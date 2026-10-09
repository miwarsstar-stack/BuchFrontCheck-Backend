import io
import base64
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from playwright.async_api import async_playwright
from PIL import Image, ImageDraw, ImageFont

app = FastAPI(title="Sell4More Live Grid API")
client = OpenAI()
executor = ThreadPoolExecutor(max_workers=5)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verarbeite_das_bild(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((1024, 1024), Image.LANCZOS)

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
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
                                "autor": {"type": "string"},
                                "verlag": {"type": "string"},
                                "erscheinungsjahr_schaetzung": {"type": "string"},
                                "format": {"type": "string"},
                                "ruecken_dicke_mm": {"type": "string"},
                                "cover_farbe": {"type": "string"},
                                "cover_merkmale": {"type": "string"},
                                "schrift_stil": {"type": "string"},
                                "auflage_hinweis": {"type": "string"},
                                "medium_typ": {"type": "string"},
                                "isbn_ean": {"type": "string"},
                                "ymin": {"type": "integer"},
                                "xmin": {"type": "integer"},
                                "ymax": {"type": "integer"},
                                "xmax": {"type": "integer"}
                            },
                            "required": [
                                "titel", "autor", "verlag",
                                "erscheinungsjahr_schaetzung", "format",
                                "ruecken_dicke_mm", "cover_farbe",
                                "cover_merkmale", "schrift_stil",
                                "auflage_hinweis", "medium_typ",
                                "isbn_ean",
                                "ymin", "xmin", "ymax", "xmax"
                            ],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["medien"],
                "additionalProperties": False
            }
        }
    }

    prompt = """Du bist ein Experte fuer Buecher, Medien und Ausgabenbestimmung.

Analysiere dieses Foto eines Regals oder einer Mediensammlung sehr genau.
Erkenne jedes einzelne sichtbare Medium (Buch, DVD, CD, Blu-ray, Spiel etc.).

Fuer jedes Medium erfasse folgende Informationen so praezise wie moeglich:

1. TITEL: exakt wie auf dem Ruecken/Cover geschrieben
2. AUTOR / INTERPRET / HERSTELLER: wie angegeben
3. VERLAG / LABEL / PUBLISHER: falls sichtbar, sonst leer lassen
4. ERSCHEINUNGSJAHR (Schaetzung): anhand von Design, Schriftbild, Logo-Stil, Papierfarbe
5. FORMAT: Taschenbuch, Hardcover, Grossformat, DVD, CD, Blu-ray etc.
6. RUECKENDICKE (geschaetzt in mm): wichtig fuer Ausgabenerkennung
7. COVER-FARBEN: dominante Farben des Rueckens/Covers
8. BESONDERE MERKMALE: Praegungen, Folienveredelung, Sticker, Sonderausgabe-Aufdruck,
   Jubilaemsedition, Filmtie-in-Cover, Buchclub-Ausgabe etc.
9. SCHRIFTSTIL: Charakteristik der Titelschrift (Farbe, Stil, Groesse)
10. AUFLAGEN-HINWEIS: sichtbare Hinweise auf Auflage oder Edition
11. MEDIUM-TYP: Buch, DVD, CD, Blu-ray, Spiel, Sonstiges
12. ISBN / EAN: Bestimme anhand deines Wissens ueber dieses Werk die wahrscheinlichste
    ISBN-13 (beginnt mit 978 oder 979) oder EAN. Nutze dabei Titel, Autor, Verlag, Format,
    Erscheinungsjahr UND alle visuellen Merkmale (Farben, Schrift, Cover-Design) um die
    genaue Ausgabe zu identifizieren. Gib NUR die Ziffern ohne Bindestriche an.
    Falls du dir nicht sicher bist, gib einen leeren String "" zurueck — KEINE Erfindung.

Gib fuer jedes Medium die exakten Pixelkoordinaten: xmin/ymin (oben links), xmax/ymax (unten rechts).
Ueberspringe kein Medium, auch wenn es schwer lesbar ist."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }],
        response_format=response_format
    )

    daten = json.loads(response.choices[0].message.content)
    print("GPT Bilderkennung:", json.dumps(daten, ensure_ascii=False, indent=2))

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=24)
    except Exception:
        font = ImageFont.load_default()

    return image, daten, draw, font


async def query_bonavendi(identifier: str) -> tuple[float, str]:
    id_clean = identifier.replace("-", "").strip()
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
    image, daten, draw, font = verarbeite_das_bild(image_bytes)

    medien = daten.get("medien", [])

    bonavendi_tasks = []
    for m in medien:
        ident = m.get("isbn_ean", "").strip()
        cleaned = "".join(c for c in ident if c.isdigit())
        if cleaned and 8 <= len(cleaned) <= 13:
            print(f"ISBN/EAN von GPT-4o fuer '{m['titel']}': {cleaned}")
            bonavendi_tasks.append(query_bonavendi(cleaned))
        else:
            if ident:
                print(f"Ungueltiges ISBN-Format von GPT fuer '{m['titel']}': '{ident}' — uebersprungen")
            else:
                print(f"Keine ISBN von GPT fuer '{m['titel']}'")
            bonavendi_tasks.append(kein_ankauf())

    preise = await asyncio.gather(*bonavendi_tasks, return_exceptions=True)

    for i, medium in enumerate(medien):
        ident = "".join(c for c in medium.get("isbn_ean", "") if c.isdigit())
        result = preise[i]
        preis, anbieter = result if not isinstance(result, Exception) else (0.0, "Kein Ankauf")

        print(
            f"Medium: {medium['titel']} | Code: {ident or '-'} | "
            f"Typ: {medium.get('medium_typ','?')} | {preis}\u20ac bei {anbieter}"
        )

        farbe_box = "#00FF00" if preis > 2.0 else ("#FFFF00" if preis > 0.0 else "#FF0000")
        zeile1 = f"{anbieter}: {preis:.2f}\u20ac" if preis > 0.0 else "Kein Ankauf"
        zeile2 = f"{ident}" if ident else "ISBN: unbekannt"
        zeile3 = f"{medium.get('format','?')} | ca. {medium.get('erscheinungsjahr_schaetzung','?')}"

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

        text_pos3 = (medium["xmin"] + 5, text_bbox2[3] + 4)
        text_bbox3 = draw.textbbox(text_pos3, zeile3, font=font)
        draw.rectangle(text_bbox3, fill="black")
        draw.text(text_pos3, zeile3, fill="white", font=font)

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
