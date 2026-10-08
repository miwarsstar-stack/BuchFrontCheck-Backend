import io
import base64
import json
import requests
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

# 1. SCHRITT: Playwright-Abfrage für die Sell4More Web-App
async def query_sell4more_web(isbn: str):
    isbn_clean = isbn.replace("-", "").strip()
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://sell4more.de", timeout=15000)

            search_input = page.locator("input[placeholder*='ISBN']")
            await search_input.wait_for(state="visible", timeout=5000)
            await search_input.fill(isbn_clean)
            await search_input.press("Enter")

            await page.wait_for_timeout(2500)

            price_element = await page.locator(".best-price-selector").first.inner_text()
            vendor_element = await page.locator(".best-vendor-selector").first.get_attribute("alt")

            await browser.close()

            price_float = float(price_element.replace("€", "").replace(",", ".").strip())
            return price_float, vendor_element or "Ankäufer"
    except Exception as e:
        print(f"Fehler bei Sell4More Web-App (ISBN: {isbn_clean}): {e}")
        return 0.0, "Fehler"

# 2. SCHRITT: ISBN über Open Library suchen (kein API-Key nötig)
def get_isbn_from_title(title: str, author: str = "") -> str:
    headers = {"User-Agent": "Sell4MoreScanner/1.0 (contact@example.com)"}

    # Variante 1: Titel + Autor über Open Library Search
    try:
        params = {
            "title": title.strip(),
            "limit": 5,
            "fields": "isbn,title,author_name"
        }
        if author.strip():
            params["author"] = author.strip()

        res = requests.get(
            "https://openlibrary.org/search.json",
            params=params,
            headers=headers,
            timeout=8
        )
        if res.status_code == 200:
            docs = res.json().get("docs", [])
            for doc in docs:
                isbns = doc.get("isbn", [])
                # ISBN-13 bevorzugen (13 Stellen)
                for isbn in isbns:
                    if len(isbn) == 13 and isbn.startswith(("978", "979")):
                        print(f"ISBN gefunden (Open Library): {isbn} für '{title}'")
                        return isbn
                # Fallback: erste ISBN nehmen
                for isbn in isbns:
                    if len(isbn) == 13:
                        print(f"ISBN gefunden (Fallback): {isbn} für '{title}'")
                        return isbn
    except Exception as e:
        print(f"Open Library Fehler für '{title}': {e}")

    # Variante 2: Nur Titel ohne Autor versuchen
    if author.strip():
        try:
            res = requests.get(
                "https://openlibrary.org/search.json",
                params={"title": title.strip(), "limit": 3, "fields": "isbn,title"},
                headers=headers,
                timeout=8
            )
            if res.status_code == 200:
                docs = res.json().get("docs", [])
                for doc in docs:
                    for isbn in doc.get("isbn", []):
                        if len(isbn) == 13 and isbn.startswith(("978", "979")):
                            print(f"ISBN gefunden (nur Titel): {isbn} für '{title}'")
                            return isbn
        except Exception as e:
            print(f"Open Library Fallback-Fehler für '{title}': {e}")

    print(f"Keine ISBN gefunden für '{title}'")
    return ""

# 3. SCHRITT: Das Bild-Verarbeitungs-Gehirn
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
                                "titel": {"type": "string"}, "autor": {"type": "string"},
                                "ymin": {"type": "integer"}, "xmin": {"type": "integer"},
                                "ymax": {"type": "integer"}, "xmax": {"type": "integer"}
                            },
                            "required": ["titel", "autor", "ymin", "xmin", "ymax", "xmax"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["buecher"], "additionalProperties": False
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

# 4. SCHRITT: BEIDE ENDPOINTS NEHMEN BILDER AN

@app.post("/")
async def scan_regal_root(file: UploadFile = File(...)):
    """Nimmt das Bild an, falls das Handy stur auf die Startseite postet"""
    image_bytes = await file.read()
    image, daten, draw, font = verarbeite_das_bild(image_bytes)

    for buch in daten["buecher"]:
        isbn = get_isbn_from_title(buch["titel"], buch["autor"])
        preis, anbieter = await query_sell4more_web(isbn) if isbn else (0.0, "Kein Ankauf")

        print(f"Buch: {buch['titel']} | ISBN: {isbn or 'nicht gefunden'} | Preis: {preis}€ | Anbieter: {anbieter}")

        farbe = "#00FF00" if preis > 2.0 else ("#FFFF00" if preis > 0.0 else "#FF0000")

        # Zeile 1: Preis oder Status
        zeile1 = f"{anbieter}: {preis:.2f}€" if preis > 0.0 else "Kein Ankauf"

        # Zeile 2: ISBN immer anzeigen
        zeile2 = f"ISBN: {isbn}" if isbn else "ISBN: -"

        box = (buch["xmin"], buch["ymin"], buch["xmax"], buch["ymax"])
        draw.rectangle(box, outline=farbe, width=6)

        # Erste Zeile zeichnen
        text_pos1 = (buch["xmin"] + 5, buch["ymin"] + 5)
        text_bbox1 = draw.textbbox(text_pos1, zeile1, font=font)
        draw.rectangle(text_bbox1, fill="black")
        draw.text(text_pos1, zeile1, fill="white", font=font)

        # Zweite Zeile direkt darunter
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
    """Klassischer Endpoint als Backup"""
    return await scan_regal_root(file)

@app.get("/")
async def health_check():
    """Zeigt an, ob der Server wach ist"""
    return {"status": "online", "info": "Sende dein Bild per POST direkt hierhin!"}

@app.head("/")
async def head_fallback():
    return {}
