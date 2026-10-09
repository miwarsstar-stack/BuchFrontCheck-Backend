import io
import base64
import json
import asyncio
import re
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from playwright.async_api import async_playwright
from PIL import Image, ImageDraw, ImageFont

app = FastAPI(title="Sell4More Live Grid API")
client = OpenAI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verarbeite_das_bild(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=90)
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "regal_erkennung",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "medien": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "titel": {"type": "string"},
                                "typ": {"type": "string", "enum": ["buch", "cd", "dvd", "spiel", "sonstiges"]},
                                "autor": {"type": "string"},
                                "verlag": {"type": "string"},
                                "ausgabe": {"type": "string"},
                                "isbn_ean": {"type": "string"},
                                "isbn_quelle": {"type": "string"},
                                "isbn_begruendung": {"type": "string"},
                                "ymin": {"type": "integer"},
                                "xmin": {"type": "integer"},
                                "ymax": {"type": "integer"},
                                "xmax": {"type": "integer"}
                            },
                            "required": ["titel", "typ", "autor", "verlag", "ausgabe", "isbn_ean", "isbn_quelle", "isbn_begruendung", "ymin", "xmin", "ymax", "xmax"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["medien"],
                "additionalProperties": False
            }
        }
    }

    prompt = """Analysiere dieses Foto und erkenne jedes sichtbare Medium (Buch, CD, DVD, Spiel usw.).

Fuer jedes Medium liefere:
- titel: Exakter Titel wie auf dem Medium sichtbar
- typ: buch, cd, dvd, spiel oder sonstiges
- autor: Sichtbarer Autor, sonst leer
- verlag: Sichtbarer Verlag, sonst leer
- ausgabe: Erkenne die wahrscheinliche konkrete Ausgabe anhand sichtbarer Merkmale, insbesondere Taschenbuch/Hardcover, Einband, Ruecken-/Covergestaltung, Sprache und Verlag. Nicht nur leere Ausgabe melden, weil kein Barcode sichtbar ist.
- isbn_ean: Wenn ein Barcode lesbar ist, lies ihn ab. Andernfalls ermittle die ISBN durch Websuche anhand von Titel + Autor und gleiche die gefundenen Ausgaben mit Sprache, Verlag, Format und sichtbaren Cover-/Rueckenmerkmalen des Fotos ab. Bevorzuge Verlagsseiten und verlaessliche Buchkataloge. Bei mehreren Ausgaben waehle die am besten zum Foto passende ISBN; nur leer lassen, wenn sich keine passende Ausgabe mit ausreichender Sicherheit bestimmen laesst. Nur Ziffern ohne Bindestriche.
- isbn_quelle: URL einer Webquelle, die ISBN und Ausgabe belegt; sonst leer.
- isbn_begruendung: Kurzer Abgleich, warum die Ausgabe passt, z. B. Taschenbuch, Verlag, Sprache oder Covermerkmale; Unsicherheit knapp benennen.
- xmin, ymin, xmax, ymax: Pixelkoordinaten des Mediums auf dem Foto"""

    response = client.responses.create(
        model="gpt-6-luna",
        tools=[{"type": "web_search"}],
        tool_choice="required",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}
            ]
        }],
        text={"format": {
            "type": "json_schema",
            "name": "regal_erkennung",
            "strict": True,
            "schema": response_format["json_schema"]["schema"]
        }}
    )

    daten = json.loads(response.output_text)
    print("GPT Erkennung mit Websuche:", json.dumps(daten, ensure_ascii=False, indent=2))

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=24)
    except Exception:
        font = ImageFont.load_default()

    return image, daten, draw, font


def _normalisiere_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").casefold()))


def _isbn13_gueltig(isbn: str) -> bool:
    if not re.fullmatch(r"(?:978|979)\d{10}", isbn):
        return False
    checksum = sum(
        int(digit) * (1 if index % 2 == 0 else 3)
        for index, digit in enumerate(isbn[:12])
    )
    return (10 - checksum % 10) % 10 == int(isbn[-1])


def suche_isbn_im_katalog(medium: dict) -> str:
    """Liefert nur eine eindeutige ISBN-13 fuer einen passenden Titel/Autor."""
    title = (medium.get("titel") or "").strip()
    author = (medium.get("autor") or "").strip()
    if len(title) < 3:
        print("Open Library: Titel fuer ISBN-Suche zu kurz.")
        return ""

    params = {
        "title": title,
        "fields": "title,author_name,isbn_13",
        "limit": "10",
    }
    if author:
        params["author"] = author
    url = "https://openlibrary.org/search.json?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url, headers={"User-Agent": "ScanSell/1.0 (ISBN lookup)"}
    )

    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            results = json.load(response).get("docs", [])
    except Exception as exc:
        print(f"Open Library Suche fehlgeschlagen fuer '{title}': {exc}")
        return ""

    title_norm = _normalisiere_text(title)
    author_norm = _normalisiere_text(author)
    candidates = set()
    for result in results:
        found_title = _normalisiere_text(result.get("title", ""))
        if not found_title or SequenceMatcher(None, title_norm, found_title).ratio() < 0.88:
            continue
        authors = result.get("author_name", [])
        if author_norm:
            if not authors:
                continue
            best_author_match = max(
                SequenceMatcher(None, author_norm, _normalisiere_text(name)).ratio()
                for name in authors
            )
            if best_author_match < 0.75:
                continue
        for isbn in result.get("isbn_13", []):
            digits = re.sub(r"\D", "", isbn)
            if _isbn13_gueltig(digits):
                candidates.add(digits)

    # Multiple ISBNs usually mean multiple editions; a spine photo cannot
    # reliably identify which one is present, so do not guess.
    print(
        f"Open Library fuer '{title}': {len(results)} Suchtreffer, "
        f"{len(candidates)} passende ISBN-13-Kandidaten."
    )
    return next(iter(candidates)) if len(candidates) == 1 else ""


async def query_bonavendi(identifier: str) -> tuple[float, str]:
    id_clean = identifier.strip()
    if not id_clean:
        return 0.0, "Kein Ankauf"
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15"
            )
            page = await context.new_page()

            api_data = []

            async def handle_response(response):
                if "bonavendi" in response.url and response.status == 200:
                    content_type = response.headers.get("content-type", "")
                    if "json" in content_type:
                        try:
                            body = await response.json()
                            api_data.append(body)
                            print(f"Bonavendi API Response: {str(body)[:300]}")
                        except Exception:
                            pass

            page.on("response", handle_response)
            await page.goto(
                f"https://www.bonavendi.de/verkaufen/?ean={id_clean}",
                timeout=20000
            )
            await page.wait_for_timeout(4000)

            bester_preis = 0.0
            bester_anbieter = "Kein Ankauf"

            for data in api_data:
                angebote = []
                if isinstance(data, list):
                    angebote = data
                elif isinstance(data, dict):
                    for key in ("offers", "results", "data", "items"):
                        if data.get(key):
                            angebote = data[key]
                            break

                for a in angebote:
                    for price_key in ("price", "buyPrice", "buy_price", "ankaufspreis", "value"):
                        val = a.get(price_key)
                        if val:
                            try:
                                p_val = float(
                                    str(val).replace(",", ".").replace("\u20ac", "").strip()
                                )
                                if p_val > bester_preis:
                                    bester_preis = p_val
                                    bester_anbieter = (
                                        a.get("vendor") or a.get("name") or
                                        a.get("shop") or "Ankaeufer"
                                    )
                            except Exception:
                                pass

            await browser.close()

            if bester_preis > 0:
                print(f"Bonavendi: {bester_preis}\u20ac bei {bester_anbieter}")
                return bester_preis, bester_anbieter

    except Exception as e:
        print(f"Bonavendi Fehler fuer {id_clean}: {e}")

    return 0.0, "Kein Ankauf"


async def kein_ankauf() -> tuple[float, str]:
    return 0.0, "Kein Ankauf"


@app.post("/")
async def scan_regal_root(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image, daten, draw, font = await asyncio.to_thread(verarbeite_das_bild, image_bytes)

    medien = daten.get("medien", [])

    bonavendi_tasks = []
    for m in medien:
        ident = re.sub(r"\D", "", m.get("isbn_ean", "").strip())
        m["isbn_ean"] = ident
        if not ident and m.get("typ") == "buch":
            ident = await asyncio.to_thread(suche_isbn_im_katalog, m)
            m["isbn_ean"] = ident
        if ident:
            print(
                f"ISBN/EAN fuer '{m['titel']}' ({m.get('ausgabe') or 'Ausgabe nicht erkannt'}): "
                f"{ident} | Quelle: {m.get('isbn_quelle') or 'Open-Library-Fallback'} | "
                f"Abgleich: {m.get('isbn_begruendung') or '-'}"
            )
            bonavendi_tasks.append(query_bonavendi(ident))
        else:
            print(
                f"Keine ISBN gefunden fuer '{m['titel']}'. Webquelle: "
                f"{m.get('isbn_quelle') or '-'} | Abgleich: {m.get('isbn_begruendung') or '-'}"
            )
            bonavendi_tasks.append(kein_ankauf())

    preise = await asyncio.gather(*bonavendi_tasks, return_exceptions=True)

    for i, medium in enumerate(medien):
        ident = medium.get("isbn_ean", "").strip()
        result = preise[i]
        preis, anbieter = result if not isinstance(result, Exception) else (0.0, "Kein Ankauf")

        print(
            f"Medium: {medium['titel']} | Code: {ident or '-'} | {preis}\u20ac bei {anbieter}"
        )

        farbe_box = "#00FF00" if preis > 2.0 else ("#FFFF00" if preis > 0.0 else "#FF0000")
        zeile1 = f"{anbieter}: {preis:.2f}\u20ac" if preis > 0.0 else "Kein Ankauf"
        zeile2 = ident if ident else "ISBN: unbekannt"

        box = (medium["xmin"], medium["ymin"], medium["xmax"], medium["ymax"])
        draw.rectangle(box, outline=farbe_box, width=6)

        text_pos1 = (medium["xmin"] + 5, medium["ymin"] + 5)
        text_bbox1 = draw.textbbox(text_pos1, zeile1, font=font)
        draw.rectangle(text_bbox1, fill="black")
        draw.text(text_pos1, zeile1, fill="white", font=font)

        text_pos2 = (medium["xmin"] + 5, text_bbox1[3] + 4)
        text_bbox2 = draw.textbbox(text_pos2, zeile2, font=font)
        draw.rectangle(text_bbox2, fill="black")
        draw.text(text_pos2, zeile2, fill="white", font=font)

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return StreamingResponse(img_byte_arr, media_type="image/jpeg")


@app.post("/scan-regal")
async def scan_regal_endpoint(file: UploadFile = File(...)):
    return await scan_regal_root(file)


@app.get("/")
async def health_check():
    return {"status": "online", "info": "Sende dein Bild per POST direkt hierhin!"}


@app.head("/")
async def head_fallback():
    return {}
