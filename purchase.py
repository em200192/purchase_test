import streamlit as st
import json
from io import BytesIO
from PIL import Image
import google.generativeai as genai
from google.api_core import exceptions
import time
import os
import re
from typing import Any, Dict, List, Optional
import pandas as pd

# ==========================================
# 🧾 Invoice Extractor – Gemini 1.5 (dotenv, RTL, Editable + Vendor Memory)
# - Uses ONLY dotenv env var GEMINI_API_KEY (no st.secrets)
# - Arabic schema keys (compatible with your current data model)
# - User can EDIT extracted lines; saves corrections to vendor_corrections.jsonl
# - "Vendor Memory": re-run with few-shot guidance learned from past corrections
# - Auto-detect/sanitize numbers; prevent size tokens as codes; basic swap-fix via memory
# ==========================================

st.set_page_config(layout="wide", page_title="🧾 Invoice Extractor – Vendor Memory")



# ---------- helpers ----------
ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789..")
CAPACITY_TOKENS = re.compile(r"(?i)\b(?:ml|ltrs?|ltr|gms?|gm|grams?|جم|غ|مل|لتر|كجم|kg|كيلو)\b")
MEMORY_PATH = "vendor_corrections.jsonl"


model_name = "gemini-2.0-flash"
def normalize_number(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().translate(ARABIC_INDIC)
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[\s\$£€¥ر.سج.د]*", "", s)
    s = s.replace(",", "")
    s = re.sub(r"\.(?=.*\.)", "", s)
    try:
        return float(s)
    except Exception:
        return None


def coerce_nulls(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: coerce_nulls(v) for k, v in x.items()}
    if isinstance(x, list):
        return [coerce_nulls(v) for v in x]
    if isinstance(x, str) and x.strip() == "":
        return None
    return x


def normalize_item_code(code: Optional[str]) -> Optional[str]:
    if not code:
        return code
    s = re.sub(r"\s+", " ", str(code)).strip()
    s = re.sub(r"^-+", "", s)
    s = " ".join(p for p in s.split() if not CAPACITY_TOKENS.search(p)).strip()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\- ]*", s):
        return s or None
    return s or None


def as_float_if_numeric_str(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        try:
            return float(s)
        except Exception:
            return None
    return None

def fill_after_tax_if_missing(item: dict, vat_rate: Optional[float] = None) -> None:
    # reserved for future use if you want to auto-compute TOTAL_AFTR_TAX using a global VAT
    if vat_rate is None:
        return
    if item.get("TOTAL_AFTR_TAX") is None and isinstance(item.get("TOTAL_BFR_TAX"), (int, float)):
        before = float(item["TOTAL_BFR_TAX"])
        item["TOTAL_AFTR_TAX"] = round(before * (1.0 + vat_rate), 2)

def validate_and_fix_schema(data: dict) -> dict:
    """
    Normalize whatever the model returns (Arabic keys or partial) into the target schema:
    VNDR_NM, CSTMR_NM, DOC_NO, DOC_NO_TAX, ITEMS[ ITM_* ].
    Also apply an arithmetic swap-fix for ITM_CODE ↔ ITM_QTY when needed.
    """
    # Top-level mapping (accept Arabic or English keys)
    vndr = data.get("VNDR_NM") or data.get("اسم_المورد")
    cstm = data.get("CSTMR_NM") or data.get("اسم_العميل")
    doc  = data.get("DOC_NO") or data.get("رقم_الفاتورة")
    doct = data.get("DOC_NO_TAX") or data.get("رقم_الفاتورة_الضريبية")
    items = data.get("ITEMS") or data.get("الأصناف")

    if not isinstance(items, list):
        items = [] if items is None else [items]

    fixed_items: List[Dict[str, Any]] = []
    for item in items:
        item = item or {}
        code         = item.get("ITM_CODE")           or item.get("رقم_الصنف")
        name_ar      = item.get("ITM_L_NM")           or item.get("اسم_الصنف")
        name_en      = item.get("ITM_F_NM")           or item.get("اسم_الصنف_انجليزي")
        unit         = item.get("ITM_UNT")            or item.get("الوحدة")
        qty          = item.get("ITM_QTY")            or item.get("الكمية")
        price        = item.get("ITM_PRICE")          or item.get("سعر_الوحدة")
        total_before = item.get("TOTAL_BFR_TAX")      or item.get("الاجمالي_قبل_الضريبة")
        disc         = item.get("ITM_DSCNT")          or item.get("الخصم")
        total_after  = item.get("TOTAL_AFTR_TAX")     or item.get("الاجمالي_بعد_الضريبة")

        fixed = {
            "ITM_CODE":        normalize_item_code(code),
            "ITM_L_NM":        name_ar,
            "ITM_F_NM":        name_en,
            "ITM_UNT":         unit,
            "ITM_QTY":         normalize_number(qty),
            "ITM_PRICE":       normalize_number(price),
            "TOTAL_BFR_TAX":   normalize_number(total_before),
            "ITM_DSCNT":       normalize_number(disc) or 0.0,
            "TOTAL_AFTR_TAX":  normalize_number(total_after),
        }

        # Arithmetic swap-fix (detect if QTY and CODE were swapped)
        qty_v      = fixed["ITM_QTY"]
        code_str   = fixed["ITM_CODE"]
        code_num   = as_float_if_numeric_str(code_str)
        unit_price = fixed["ITM_PRICE"]
        line_total = fixed["TOTAL_BFR_TAX"]

        if unit_price and line_total and unit_price != 0:
            expected_qty = line_total / unit_price
            eps = max(0.02, abs(expected_qty) * 0.01)  # ±1% or 0.02
            def close(a, b): return (a is not None and b is not None and abs(a - b) <= eps)
            qty_matches  = close(qty_v, expected_qty)
            code_matches = close(code_num, expected_qty)

            if (not qty_matches) and code_matches:
                prev_qty = qty_v
                fixed["ITM_QTY"] = code_num
                if isinstance(prev_qty, (int, float)) and abs(prev_qty - round(prev_qty)) < 1e-6 and 0 < prev_qty <= 1_000_000:
                    fixed["ITM_CODE"] = str(int(round(prev_qty)))
                else:
                    fixed["ITM_CODE"] = code_str if code_str and not re.fullmatch(r"\d+(?:\.\d+)?", code_str) else None

        # (optional) fill_after_tax_if_missing(fixed, vat_rate=None)
        fixed_items.append(fixed)

    out = {
        "VNDR_NM":   vndr,
        "CSTMR_NM":  cstm,
        "DOC_NO":    doc,
        "DOC_NO_TAX": doct,
        "ITEMS":     fixed_items,
    }
    return coerce_nulls(out)



# ---------- lightweight vendor memory ----------

def load_memory() -> List[dict]:
    if not os.path.exists(MEMORY_PATH):
        return []
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception:
        return []


def save_memory(record: dict) -> None:
    try:
        with open(MEMORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def get_vendor_examples(vendor: str, max_n: int = 3) -> List[dict]:
    mem = load_memory()
    examples = [r for r in mem if (r.get("اسم_المورد") == vendor or r.get("VNDR_NM") == vendor)]
    return examples[:max_n]


def build_vendor_hint(vendor: str) -> str:
    ex = get_vendor_examples(vendor)
    if not ex:
        return ""
    lines = ["\n\n📌 Vendor-specific guidance (learned from previous corrections):", f"- Vendor: {vendor}"]
    for i, r in enumerate(ex, start=1):
        items = r.get("الأصناف") or r.get("ITEMS") or []
        if not items:
            continue
        s = items[0]
        code = s.get("رقم_الصنف") or s.get("ITM_CODE")
        qty = s.get("الكمية") or s.get("ITM_QTY")
        lines.append(f"  • Example {i}: item_code='{code}', qty={qty}  → code from rightmost code column; qty from QTY column")
    lines.append("- Never use size tokens (e.g., 230ML, 160 Gms) as codes. If unit_price*qty mismatches but unit_price*code matches the line total, swap them.")
    return "\n".join(lines)


# ---------- Gemini ----------

def get_api_key() -> Optional[str]:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_model(model_name: str = model_name, enforce_json: bool = True):
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment (.env)")
    genai.configure(api_key=api_key)
    generation_config = {"response_mime_type": "application/json"} if enforce_json else {}
    return genai.GenerativeModel(model_name=model_name, generation_config=generation_config)


def image_to_jpeg_bytes(img: Image.Image, max_side: int = 2400, quality: int = 92) -> bytes:
    try:
        img = img.convert("RGB")
    except Exception:
        img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    buf = BytesIO()
    img.save(buf, format="JPEG", optimize=True, quality=quality)
    return buf.getvalue()


def call_gemini_api(image_bytes: bytes, prompt: str, model_name: str) -> tuple[Optional[dict], Optional[str]]:
    if not image_bytes:
        return None, "لا توجد صورة"
    try:
        model = get_model(model_name, enforce_json=True)
    except Exception as e:
        return None, f"تهيئة فشلت: {e}"

    last_err = None
    delay = 2
    for attempt in range(5):
        try:
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            resp = model.generate_content([prompt, img])
            text = getattr(resp, "text", "") or ""
            try:
                data = json.loads(text)
            except Exception:
                # try to recover first {...}
                data = None
                stack, start = [], None
                for i, ch in enumerate(text):
                    if ch == '{':
                        if not stack:
                            start = i
                        stack.append(ch)
                    elif ch == '}':
                        if stack:
                            stack.pop()
                            if not stack and start is not None:
                                cand = text[start:i+1]
                                try:
                                    data = json.loads(cand)
                                    break
                                except Exception:
                                    start = None
                                    continue
            if not data:
                raise ValueError("JSON غير صالح")
            return data, None
        except exceptions.ResourceExhausted as e:
            last_err = str(e)
            if attempt < 4:
                time.sleep(delay)
                delay *= 2
            else:
                return None, "تجاوز حد الاستخدام كثيرًا"
        except Exception as e:
            last_err = str(e)
            if attempt < 4:
                time.sleep(delay)
                delay *= 2
            else:
                return None, f"خطأ: {e}"
    return None, last_err or "فشل غير معروف"


USER_PROMPT = r"""
**CRITICAL TASK: Parse the invoice image and return ONLY a valid JSON (UTF-8) matching EXACTLY this schema and keys.**

Top-level keys:
- "VNDR_NM": string|null  (Vendor name)
- "CSTMR_NM": string|null (Customer name)
- "DOC_NO": string|null   (Invoice number)
- "DOC_NO_TAX": string|null (Tax invoice number; if not printed, return null)
- "ITEMS": [ { line objects as below } ]

For each line in ITEMS (read the item table rows):
- "ITM_CODE": string|null  → Value from the Item Code column ONLY (Arabic RTL invoices often put this on the FAR RIGHT). It may be a simple integer like 6, 14, 28 OR a dashed code like 09-001-010. Never take numbers from the description or from QTY. If the code wraps across lines, join the parts while keeping dashes/spaces.
- "ITM_L_NM": string|null  → Arabic description ONLY
- "ITM_F_NM": string|null  → English description for the same row ONLY
- "ITM_UNT": string|null   → Unit, only if there is a dedicated unit column; else null
- "ITM_QTY": number|null   → Quantity from QTY/الكمية column ONLY
- "ITM_PRICE": number|null → Unit price
- "TOTAL_BFR_TAX": number|null → Line total before VAT/tax
- "ITM_DSCNT": number      → Discount amount for the line; 0 if none
- "TOTAL_AFTR_TAX": number|null → Line total after VAT/tax; if not printed per-line, return null.

STRICT RULES:
1) Read Arabic invoices from RIGHT to LEFT. The Item Code column is often the rightmost column. Do NOT confuse it with the counter (#) or QTY.
2) Do NOT use size/capacity tokens from description as codes (e.g., 230ML, 160 Gms, 1 Ltrs, 230 جم). Those are not item codes.
3) All numeric values must be JSON numbers (not strings). Keep decimal precision as printed.
4) If a field is not present, return null.
5) Self-check: if (ITM_PRICE × ITM_QTY) does NOT approximately equal TOTAL_BFR_TAX (±1%), BUT (ITM_PRICE × ITM_CODE_as_number_if_numeric) DOES ≈ TOTAL_BFR_TAX, then you swapped QTY and CODE; fix it so that QTY holds the value satisfying the equation, and CODE holds the other value.

Return ONLY the JSON object, nothing else.
"""



# ---------- UI ----------
with st.sidebar:
    api_key = st.secrets.get("GEMINI_API_KEY", None)

    use_memory = st.checkbox("🧠 استخدام ذاكرة الموردين عند الإعادة", value=True)

st.title("🧾 مستخرج بيانات الفواتير – قابل للتعديل مع ذاكرة الموردين")
st.markdown("---")

# upload
try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except Exception:
    PDF_SUPPORT = False
    st.warning("لم يتم تثبيت pdf2image – لن تُحوّل ملفات PDF.")

files = st.file_uploader("ارفع فواتير (PDF/JPG/PNG)", type=["pdf","jpg","jpeg","png"], accept_multiple_files=True)

results: List[Dict[str, Any]] = []
previews: List[Dict[str, Any]] = []

if files:
    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.subheader("المعاينة")
        for f in files:
            if f.type == "application/pdf" and PDF_SUPPORT:
                pages = convert_from_bytes(f.getvalue(), dpi=200)
                for i, p in enumerate(pages, 1):
                    previews.append({"name": f"{f.name} – {i}", "image": p})
                    st.image(p, caption=f"{f.name} – {i}", use_column_width=True)
            else:
                img = Image.open(f)
                previews.append({"name": f.name, "image": img})
                st.image(img, caption=f.name, use_column_width=True)

    with col2:
        st.subheader("الاستخراج")
        if st.button("🚀 استخراج", type="primary", disabled=not api_key):
            total = len(previews)
            prog = st.progress(0, text=f"تحليل {total} صفحة…")
            for i, item in enumerate(previews, 1):
                prog.progress(i/total, text=f"{i}/{total}")
                img_bytes = image_to_jpeg_bytes(item["image"], max_side=2400)

                # خطوة 1: استخراج أساسي
                raw1, err1 = call_gemini_api(img_bytes, USER_PROMPT, model_name)
                with st.expander(f"📄 {item['name']}", expanded=True):
                    if not raw1:
                        st.error(err1)
                        continue
                    fixed1 = validate_and_fix_schema(raw1)

                    # محرر قابل للتعديل
                    vendor = fixed1.get("اسم_المورد") or ""
                    st.markdown(f"**المورد:** {vendor}")
                    df = pd.DataFrame(fixed1.get("الأصناف", []))
                    edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic", key=f"edit_{i}")

                    colA, colB, colC = st.columns(3)
                    with colA:
                        if st.button("💾 حفظ التصحيحات في الذاكرة", key=f"save_{i}"):
                            fixed1["الأصناف"] = edited_df.fillna(value=None).to_dict(orient="records")
                            save_memory({"اسم_المورد": vendor, **fixed1})
                            st.success("تم الحفظ. سيتم استخدام هذه التصحيحات كأمثلة مستقبلية.")
                    with colB:
                        st.download_button("⬇️ تنزيل JSON", data=json.dumps(fixed1, ensure_ascii=False).encode("utf-8"), file_name=f"{item['name']}.json", mime="application/json", key=f"dl_{i}")
                    with colC:
                        if use_memory and vendor and st.button("🔁 إعادة الاستخراج باستخدام ذاكرة المورد", key=f"rerun_{i}"):
                            hint = build_vendor_hint(vendor)
                            prompt2 = USER_PROMPT + ("\n" + hint if hint else "")
                            raw2, err2 = call_gemini_api(img_bytes, prompt2, model_name)
                            if raw2:
                                fixed2 = validate_and_fix_schema(raw2)
                                st.markdown("**نتيجة بعد استخدام الذاكرة:**")
                                st.json(fixed2)
                                # اقتراح: يمكن مقارنة الفروقات هنا إذا رغبت
                            else:
                                st.warning(f"فشل إعادة الاستخراج: {err2}")

                    st.markdown("**النتيجة الحالية:**")
                    st.json(fixed1)
                    results.append({"file": item["name"], "data": fixed1})

            prog.empty()
            st.success("اكتمل الاستخراج")

        # عرض الذاكرة الحالية اختياريًا
        with st.expander("🧠 ذاكرة الموردين المخزنة"):
            mem = load_memory()
            st.write(f"{len(mem)} سجل/سجلات مصححة محفوظة")
            if mem:
                st.dataframe(pd.DataFrame([{
                    "المورد": r.get("اسم_المورد") or r.get("VNDR_NM"),
                    "رقم الفاتورة": r.get("رقم_الفاتورة") or r.get("DOC_NO"),
                    "عدد الأصناف": len(r.get("الأصناف") or r.get("ITEMS") or [])
                } for r in mem]), use_container_width=True)

# ===== end of file =====
