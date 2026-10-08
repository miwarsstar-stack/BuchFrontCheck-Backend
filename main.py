import io
import base64
import json
import asyncio
import requests
import aiohttp
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
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


# 1. SCHRITT: Ankaufpreise direkt per HTTP (kein Playwright)
async def _query_momox(session: aiohttp.ClientSession, isbn: str) -> tuple[float, str]:
    try:
        url = f"https://www.momox.de/api/sell/books/{isbn}"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as r:
            if r.status == 200:
                data = await r.json(content_type=None)
                price = float(data.get("sellPrice") or data.get("price") or 0)
                if price > 0:
                    return price, "Momox"
    except Exception as e:
        print(f"Momox Fehler {isbn}: {e}")
    return 0.0, "Momox"


async def _query_rebuy(session: aiohttp.ClientSession, isbn: str) -> tuple[float, str]:
    try:
        url = f"https://www.rebuy.de/api/v3/buy-orders/calculate?ean={isbn}"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as r:
            if r.status == 200:
                data = await r.json(content_type=None)
                price = float(data.get("price") or data.get("totalPrice") or 0)
                if price > 0:
                    return price, "reBuy"
    except Exception as e:
        print(f"reBuy Fehler {isbn}: {e}")
    return 0.0, "reBuy"


async def _query_studibuch(session: aiohttp.ClientSession, isbn: str) -> tuple[float, str]:
    try:
        url = f"https://www.studibuch.de/api/sell?isbn={isbn}"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as r:
            if r.status == 200:
                data = await r.json(content_type=None)
                price = float(data.get("price") or data.get("buyPrice") or 0)
                if price > 0:
                    return price, "Studibuch"
    except Exception as e:
        print(f"Studibuch Fehler {isbn}: {e}")
    return 0.0, "Studibuch"


async def query_bonavendi(isbn: str) -> tuple[float, str]:
    isbn_clean = isbn.replace("-", "").strip()
    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
        "Accept": "application/json",
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [
            _query_momox(session, isbn_clean),
            _query_rebuy(session, isbn_clean),
            _query_studibuch(session, isbn_clean),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    bester_preis = 0.0
    bester_anbieter = "Kein Ankauf"

    for r in results:
        if isinstance(r, Exception):
            continue
        preis, anbieter = r
        if preis > bester_preis:
            bester_preis = preis
            bester_anbieter = anbieter

    print(f"Bestes Angebot für {isbn_clean}: {bester_preis}€ bei {bester_anbieter}")
    return bester_preis, bester_anbieter


# 2. SCHRITT: Titel normalisieren
def normalize_title(title: str) -> str:
    if title == title.upper():
        return title.title()
    return title


# 3. SCHRITT: ISBN über Open Library suchen
def get_isbn_from_title(title: str, author: str = "") -> str:
    headers = {"User-Agent": "Sell4MoreScanner/1.0"}
    title_norm = normalize_title(title.strip())
    author_norm = normalize_title(author.strip())

    search_variants = []
    if author_norm:
        search_variants.append({"title": title_norm, "author": author_norm})
    search_variants.append({"title": title_norm})
    if title_norm != title.strip():
        search_variants.append({"title": title.strip()})
    short_title = " ".join(title_norm.split()[:3])
    if short_title != title_norm:
        search_variants.append({"title": short_title})

    for params in search_variants:
        try:
            params["limit"] = 5
            params["fields"] = "isbn,title,author_name"
            res = requests.get(
                "https://openlibrary.org/search.json",
                params=params,
                headers=headers,
                timeout=8
            )
            if res.status_code == 200:
                docs = res.json().get("docs", [])
                for doc in docs:
                    for isbn in doc.get("isbn", []):
                        if len(isbn) == 13 and isbn.startswith(("978", "979")):
                            print(f"ISBN gefunden: {isbn} für '{title}'")
                            return isbn
        except Exception as e:
            print(f"Open Library Fehler: {e}")

    print(f"Keine ISBN gefunden für '{title}'")
    return ""


# 4. SCHRITT: Bild verarbeiten mit GPT-4o
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


# 5. SCHRITT: ENDPOINTS

@app.post("/")
async def scan_regal_root(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image, daten, draw, font = verarbeite_das_bild(image_bytes)

    for buch in daten["buecher"]:
        isbn = get_isbn_from_title(buch["titel"], buch["autor"])
        preis, anbieter = await query_bonavendi(isbn) if isbn else (0.0, "Kein Ankauf")

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
