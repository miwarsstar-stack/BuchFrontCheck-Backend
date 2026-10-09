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


# 1. ISBN per GPT-4o-search-preview (live Websuche)
def get_isbn_via_search(titel: str, autor: str) -> str:
    try:
        query = f"ISBN-13 des Buches \"{titel}\" von \"{autor}\""
        if not autor:
            query = f"ISBN-13 des Buches \"{titel}\""
        response = client.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[{"role": "user", "content": 
                f"Was ist die ISBN-13 des Buches '{titel}' von '{autor}'? "
                f"Antworte NUR mit der 13-stelligen ISBN-Zahl, ohne Text davor oder danach. "
                f"Falls du keine eindeutige ISBN findest, antworte mit leer."
            }]
        )
        result = response.choices[0].message.content.strip()
        # Nur Ziffern extrahieren
        digits = "".join(c for c in result if c.isdigit())
        if len(digits) == 13 and digits.startswith(("978", "979")):
            print(f"ISBN gefunden: {digits} für '{titel}'")
            return digits
        print(f"Keine gültige ISBN für '{titel}': {result}")
    except Exception as e:
        print(f"ISBN-Suche Fehler für '{titel}': {e}")
    return ""


# 2. Bonavendi per Playwright
async def query_bonavendi(isbn: str) -> tuple[float, str]:
    isbn_clean = isbn.replace("-", "").strip()
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
                        except:
                            pass

            page.on("response", handle_response)
            await page.goto(f"https://www.bonavendi.de/verkaufen/?ean={isbn_clean}", timeout=20000)
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
                                p_val = float(str(val).replace(",", ".").replace("€", "").strip())
                                if p_val > bester_preis:
                                    bester_preis = p_val
                                    bester_anbieter = a.get("vendor") or a.get("name") or a.get("shop") or "Ankäufer"
                            except:
                                pass

            await browser.close()

            if bester_preis > 0:
                print(f"Bonavendi bestes Angebot: {bester_preis}€ bei {bester_anbieter}")
                return bester_preis, bester_anbieter

    except Exception as e:
        print(f"Bonavendi Fehler für {isbn_clean}: {e}")

    return 0.0, "Kein Ankauf"


async def kein_ankauf() -> tuple[float, str]:
    return 0.0, "Kein Ankauf"


# 3. Bild verarbeiten mit GPT-4o
def verarbeite_das_bild(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((1024, 1024), Image.LANCZOS)

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "regal_erkennung",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "buecher": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "titel": {"type": "string"},
                                "autor": {"type": "string"},
                                "ymin": {"type": "integer"},
                                "xmin": {"type": "integer"},
                                "ymax": {"type": "integer"},
                                "xmax": {"type": "integer"}
                            },
                            "required": ["titel", "autor", "ymin", "xmin", "ymax", "xmax"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["buecher"],
                "additionalProperties": False
            }
        }
    }

    prompt = """Analysiere dieses Foto eines Bücherregals sehr genau.
Erkenne jeden einzelnen sichtbaren Buchrücken.
Für jedes Buch gib mir:
- Den genauen Titel (wie auf dem Buchrücken geschrieben)
- Den Autor (wie auf dem Buchrücken geschrieben)
- Die exakten Pixelkoordinaten des Buchrückens: xmin, ymin (obere linke Ecke) und xmax, ymax (untere rechte Ecke).
Die Koordinaten müssen den Buchrücken eng und präzise umschließen. Überspringe kein Buch."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}],
        response_format=response_format
    )

    daten = json.loads(response.choices[0].message.content)
    print("GPT Antwort:", json.dumps(daten, ensure_ascii=False, indent=2))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=24)
    except:
        font = ImageFont.load_default()

    return image, daten, draw, font


# 4. ENDPOINTS

@app.post("/")
async def scan_regal_root(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image, daten, draw, font = verarbeite_das_bild(image_bytes)

    # Alle ISBN-Suchen parallel (in ThreadPool weil synchron)
    loop = asyncio.get_event_loop()
    isbn_tasks = [
        loop.run_in_executor(executor, get_isbn_via_search, b["titel"], b["autor"])
        for b in daten["buecher"]
    ]
    isbns = await asyncio.gather(*isbn_tasks)

    # Alle Bonavendi-Abfragen parallel
    bonavendi_tasks = [
        query_bonavendi(isbn) if isbn else kein_ankauf()
        for isbn in isbns
    ]
    preise = await asyncio.gather(*bonavendi_tasks, return_exceptions=True)

    for i, buch in enumerate(daten["buecher"]):
        isbn = isbns[i]
        result = preise[i]
        preis, anbieter = result if not isinstance(result, Exception) else (0.0, "Kein Ankauf")

        print(f"Buch: {buch['titel']} | ISBN: {isbn or '-'} | Preis: {preis}€ | Anbieter: {anbieter}")

        farbe = "#00FF00" if preis > 2.0 else ("#FFFF00" if preis > 0.0 else "#FF0000")
        zeile1 = f"{anbieter}: {preis:.2f}€" if preis > 0.0 else "Kein Ankauf"
        zeile2 = f"ISBN: {isbn}" if isbn else "ISBN: -"

        box = (buch["xmin"], buch["ymin"], buch["xmax"], buch["ymax"])
        draw.rectangle(box, outline=farbe, width=6)

        text_pos1 = (buch["xmin"] + 5, buch["ymin"] + 5)
        text_bbox1 = draw.textbbox(text_pos1, zeile1, font=font)
        draw.rectangle(text_bbox1, fill="black")
        draw.text(text_pos1, zeile1, fill="white", font=font)

        text_pos2 = (buch["xmin"] + 5, text_bbox1[3] + 4)
        text_bbox2 = draw.textbbox(text_pos2, zeile2, font=font)
        draw.rectangle(text_bbox2, fill="black")
        draw.text(text_pos2, zeile2, fill="white", font=font)

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
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
