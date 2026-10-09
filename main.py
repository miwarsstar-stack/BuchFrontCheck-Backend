import io
import base64
import json
import asyncio
import re
import math
import time
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
BONAVENDI_ENABLED = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verarbeite_das_bild(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Keep the original for the returned image, but cap the model input size.
    model_image = image.copy()
    model_image.thumbnail((1600, 1600), Image.LANCZOS)
    scale_x = image.width / model_image.width
    scale_y = image.height / model_image.height

    buffered = io.BytesIO()
    model_image.save(buffered, format="JPEG", quality=90)
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
                                "sicherheit_prozent": {"type": "integer"},
                                "ecken": {
                                    "type": "object",
                                    "properties": {
                                        "punkt_1": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"], "additionalProperties": False},
                                        "punkt_2": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"], "additionalProperties": False},
                                        "punkt_3": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"], "additionalProperties": False},
                                        "punkt_4": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"], "additionalProperties": False}
                                    },
                                    "required": ["punkt_1", "punkt_2", "punkt_3", "punkt_4"],
                                    "additionalProperties": False
                                }
                            },
                            "required": ["titel", "typ", "autor", "verlag", "ausgabe", "sicherheit_prozent", "ecken"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["medien"],
                "additionalProperties": False
            }
        }
    }

    prompt = """Analysiere dieses Foto und erkenne jedes sichtbare Medium (Buch, CD, DVD, Spiel usw.). Dies ist der schnelle Erkennungsschritt: keine Websuche und keine ISBN-Recherche. Konzentriere dich auf genaue sichtbare Metadaten und die Koordinaten.

Fuer jedes Medium liefere:
- titel: Exakter Titel wie auf dem Medium sichtbar
- typ: buch, cd, dvd, spiel oder sonstiges
- autor: Sichtbarer Autor, sonst leer
- verlag: Sichtbarer Verlag, sonst leer
- ausgabe: Kurze Formatangabe, z. B. Hardcover, Taschenbuch mit Klappen oder gebunden. Bestimme die Ausgabe anhand des Fotos, nicht nur anhand eines Barcodes.
- sicherheit_prozent: Sicherheit der visuellen Erkennung von Titel, Autor und Format in diesem Foto.
- ecken: Vier Eckpunkte des sichtbaren Mediums als perspektivisches Viereck. Folge der sichtbaren Außenkante im Uhrzeigersinn; beginne am im Bild obersten Eckpunkt, bei Gleichstand am linkesten. Die Punkte dürfen schräg und ungleichwinklig sein. Pixelkoordinaten im gesendeten Bild (maximal 1600 Pixel an der längsten Seite)."""

    response = client.responses.create(
        model="gpt-6-luna",
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
    for medium in daten.get("medien", []):
        medium["isbn_ean"] = ""
        medium["isbn_quelle"] = ""
        medium["auflage_nicht_eindeutig"] = False
        for point in medium["ecken"].values():
            point["x"] = round(point["x"] * scale_x)
            point["y"] = round(point["y"] * scale_y)
    print("GPT-Erkennung:", json.dumps(daten, ensure_ascii=False))

    draw = ImageDraw.Draw(image)
    return image, daten, draw


def _medium_crop_base64(image: Image.Image, medium: dict) -> str:
    points = [
        (medium["ecken"][f"punkt_{index}"]["x"], medium["ecken"][f"punkt_{index}"]["y"])
        for index in range(1, 5)
    ]
    left = min(point[0] for point in points)
    top = min(point[1] for point in points)
    right = max(point[0] for point in points)
    bottom = max(point[1] for point in points)
    margin_x = max(8, int((right - left) * 0.04))
    margin_y = max(8, int((bottom - top) * 0.04))
    crop = image.crop((
        max(0, left - margin_x),
        max(0, top - margin_y),
        min(image.width, right + margin_x),
        min(image.height, bottom + margin_y),
    ))
    crop.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    crop.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _recherche_isbn_web(medium: dict, crop_base64: str) -> dict:
    started = time.perf_counter()
    schema = {
        "type": "object",
        "properties": {
            "isbn_ean": {"type": "string"},
            "isbn_quelle": {"type": "string"},
            "sicherheit_prozent": {"type": "integer"},
            "auflage_nicht_eindeutig": {"type": "boolean"},
        },
        "required": ["isbn_ean", "isbn_quelle", "sicherheit_prozent", "auflage_nicht_eindeutig"],
        "additionalProperties": False,
    }
    prompt = f"""Ermittle fuer genau dieses Buch die wahrscheinlichste passende ISBN-10, ISBN-13 oder EAN anhand des Buchfotos und Webrecherche.
Erkanntes Buch: Titel={medium.get('titel') or ''}; Autor={medium.get('autor') or ''}; Verlag={medium.get('verlag') or ''}; Ausgabe={medium.get('ausgabe') or ''}.
Vergleiche gefundene Ausgaben mit dem Foto: Coverfarben, Motiv, Typografie, Anordnung, Verlagssignet, Buchformat und Ruecken. Titel und Autor allein reichen nicht. Waehle auch bei mehreren plausiblen Kandidaten die wahrscheinlichste Variante passend zum Foto; lasse die Kennung nur leer, wenn kein plausibler Kandidat recherchierbar ist. Erfinde keine Ziffern. Bevorzuge Verlagsseiten und Kataloge, die ISBN und Coverausgabe belegen.
sicherheit_prozent ist die Sicherheit der Erkennung und der wahrscheinlichsten Kennung. Mehrere Kandidaten senken diesen Wert fuer sich allein nicht.
auflage_nicht_eindeutig ist nur ein Zusatzhinweis: Setze true ausschliesslich, wenn mindestens zwei konkrete ISBN-Ausgaben fuer denselben Titel und Autor nach Abgleich von Coverdesign, Farbe, Format, Sprache und Verlag weiterhin realistisch zu genau diesem Foto passen. Treffer zum selben Werk mit klar abweichendem Cover/Format zaehlen nicht. Bei einem eindeutigen starken Cover-Treffer setze false, selbst wenn die Suche weitere Ausgaben des Werks findet. Diese Kennzeichnung darf sicherheit_prozent nicht beeinflussen.
Gib als isbn_quelle eine verlaessliche URL fuer die ausgewaehlte Kennung und Ausgabe aus."""
    response = client.responses.create(
        model="gpt-6-luna",
        tools=[{"type": "web_search"}],
        tool_choice="required",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{crop_base64}", "detail": "high"},
            ],
        }],
        text={"format": {
            "type": "json_schema",
            "name": "isbn_ausgabe",
            "strict": True,
            "schema": schema,
        }},
    )
    print(
        f"OpenAI-Webrecherche fuer '{medium.get('titel', '')}' "
        f"dauerte {time.perf_counter() - started:.2f}s."
    )
    return json.loads(response.output_text)


def _normalisiere_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").casefold()))


def _isbn10_gueltig(isbn: str) -> bool:
    if not re.fullmatch(r"\d{9}[\dX]", isbn):
        return False
    checksum = sum(int(digit) * (10 - index) for index, digit in enumerate(isbn[:9]))
    check_value = 10 if isbn[-1] == "X" else int(isbn[-1])
    return (checksum + check_value) % 11 == 0


def _ean_gueltig(code: str) -> bool:
    if not code.isdigit() or len(code) not in (8, 12, 13, 14):
        return False
    checksum = sum(
        int(digit) * (3 if (len(code) - 2 - index) % 2 == 0 else 1)
        for index, digit in enumerate(code[:-1])
    )
    return (10 - checksum % 10) % 10 == int(code[-1])


def _isbn13_gueltig(isbn: str) -> bool:
    return len(isbn) == 13 and isbn.startswith(("978", "979")) and _ean_gueltig(isbn)


def zeichne_buchtext(image, points, medium, price_text):
    """Zeichnet ein oberes Preisfeld und ein passend skaliertes Datenfeld."""
    def load_label_font(size):
        for font_name in ("DejaVuSans-Bold.ttf", "arialbd.ttf"):
            try:
                return ImageFont.truetype(font_name, size)
            except OSError:
                continue
        return ImageFont.load_default(size=size)

    edges = [(points[index], points[(index + 1) % 4]) for index in range(4)]
    edge_lengths = [
        math.hypot(edge[1][0] - edge[0][0], edge[1][1] - edge[0][1])
        for edge in edges
    ]
    long_edge_index = max(range(4), key=lambda index: edge_lengths[index])
    long_side = min(edge_lengths[long_edge_index], edge_lengths[(long_edge_index + 2) % 4])
    short_edge_indices = ((long_edge_index + 1) % 4, (long_edge_index + 3) % 4)
    short_edge_index = min(short_edge_indices, key=lambda index: (edges[index][0][1] + edges[index][1][1]) / 2)
    short_side = min(edge_lengths[index] for index in short_edge_indices)
    padding = max(4, int(short_side * 0.035))
    center_x = sum(x for x, _ in points) / 4
    center_y = sum(y for _, y in points) / 4
    center = (center_x, center_y)
    long_start, long_end = edges[long_edge_index]
    axis_x, axis_y = long_end[0] - long_start[0], long_end[1] - long_start[1]
    if axis_x < 0 or (axis_x == 0 and axis_y < 0):
        axis_x, axis_y = -axis_x, -axis_y
    angle = math.degrees(math.atan2(axis_y, axis_x))

    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def make_panel(lines, font, line_height):
        panel_width = max(int(measure.textlength(line, font=font)) for line in lines) + padding * 2
        panel_height = len(lines) * line_height + padding * 2
        panel = Image.new("RGBA", (max(1, panel_width), max(1, panel_height)), (0, 0, 0, 0))
        panel_draw = ImageDraw.Draw(panel)
        panel_draw.rounded_rectangle(
            (0, 0, panel.width - 1, panel.height - 1),
            radius=6,
            fill=(0, 0, 0, 255),
            outline=(0, 0, 0, 255),
            width=1,
        )
        for line_index, line in enumerate(lines):
            text_width = measure.textlength(line, font=font)
            bbox = measure.textbbox((0, 0), line, font=font)
            text_y = padding + line_index * line_height - bbox[1]
            x = (panel.width - text_width) / 2
            label = next(
                (prefix for prefix in ("Titel: ", "Autor: ", "ISBN/EAN: ", "Sicherheit: ", "Preis: ") if line.startswith(prefix)),
                None,
            )
            if label:
                panel_draw.text((x, text_y), label, font=font, fill=(255, 45, 45, 255))
                x += measure.textlength(label, font=font)
                panel_draw.text((x, text_y), line[len(label):], font=font, fill=(255, 255, 255, 255))
            else:
                panel_draw.text((x, text_y), line, font=font, fill=(255, 255, 255, 255))
        return panel

    def paste_along_axis(panel, panel_center):
        rotated = panel.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC)
        position = (
            round(panel_center[0] - rotated.width / 2),
            round(panel_center[1] - rotated.height / 2),
        )
        image.paste(rotated, position, rotated)

    # Place the price field near the top end, aligned with the book's long axis.
    top_start, top_end = edges[short_edge_index]
    top_mid = ((top_start[0] + top_end[0]) / 2, (top_start[1] + top_end[1]) / 2)
    inward_x, inward_y = center_x - top_mid[0], center_y - top_mid[1]
    inward_length = max(1, math.hypot(inward_x, inward_y))
    inward_unit = (inward_x / inward_length, inward_y / inward_length)
    max_price_width = max(1, int(long_side * 0.40) - padding * 2)
    max_price_height = max(1, int(short_side - padding * 2))
    price_panel = None
    price_width = 1
    for font_size in (44, 40, 36, 32, 28, 24, 20, 16, 12, 10, 8):
        price_font = load_label_font(font_size)
        price_bbox = measure.textbbox((0, 0), price_text, font=price_font)
        price_line_height = price_bbox[3] - price_bbox[1] + 2
        candidate = make_panel([price_text], price_font, price_line_height)
        if candidate.width <= max_price_width and candidate.height <= max_price_height:
            price_panel = candidate
            break
    if price_panel is None:
        price_font = load_label_font(8)
        price_bbox = measure.textbbox((0, 0), price_text, font=price_font)
        price_panel = make_panel([price_text], price_font, price_bbox[3] - price_bbox[1] + 2)
        price_panel.thumbnail((max_price_width, max_price_height), Image.Resampling.LANCZOS)
    price_width = price_panel.width
    price_center = (
        top_mid[0] + inward_unit[0] * (padding + price_width / 2),
        top_mid[1] + inward_unit[1] * (padding + price_width / 2),
    )
    paste_along_axis(price_panel, price_center)

    # The data field uses the remaining book length and full available width.
    data_shift = int(long_side * 0.08)
    data_axis_length = max(1, int(long_side - price_width - padding * 3 - data_shift))
    try:
        confidence = max(0, min(100, int(medium.get("sicherheit_prozent", 0))))
    except (TypeError, ValueError):
        confidence = 0
    labels = [
        f"Titel: {medium.get('titel') or 'unbekannt'}",
        f"Autor: {medium.get('autor') or 'unbekannt'}",
        f"ISBN/EAN: {medium.get('isbn_ean') or 'unbekannt'}",
        f"Sicherheit: {confidence}%" + (" (Auflage nicht eindeutig)" if medium.get("auflage_nicht_eindeutig") else ""),
    ]
    content_height = max(1, int(short_side - padding * 2))
    max_line_width = data_axis_length

    fit = None
    for font_size in (48, 44, 40, 36, 32, 28, 24, 22, 20, 18, 16, 14, 12, 10, 8):
        font = load_label_font(font_size)
        lines = []
        for label in labels:
            current = ""
            for word in label.split():
                candidate = f"{current} {word}".strip()
                if current and measure.textlength(candidate, font=font) > max_line_width:
                    lines.append(current)
                    current = word
                else:
                    current = candidate
            if current:
                lines.append(current)
        bbox = measure.textbbox((0, 0), "Ag", font=font)
        line_height = max(font_size + 3, bbox[3] - bbox[1] + 4)
        panel_height = len(lines) * line_height + padding * 2
        panel_width = max(int(measure.textlength(line, font=font)) for line in lines) + padding * 2
        if panel_height <= content_height and panel_width <= data_axis_length:
            fit = (font, lines, line_height)
            break

    if fit is None:
        font = load_label_font(8)
        lines = []
        for label in labels:
            current = ""
            for word in label.split():
                candidate = f"{current} {word}".strip()
                if current and measure.textlength(candidate, font=font) > max_line_width:
                    lines.append(current)
                    current = word
                else:
                    current = candidate
            if current:
                lines.append(current)
        bbox = measure.textbbox((0, 0), "Ag", font=font)
        line_height = max(9, bbox[3] - bbox[1] + 2)
    else:
        font, lines, line_height = fit

    data_panel = make_panel(lines, font, line_height)
    data_panel.thumbnail((data_axis_length, content_height), Image.Resampling.LANCZOS)
    # Keep the data panel after the price panel along the same long axis.
    data_offset = padding * 2 + price_width + data_shift + data_panel.width / 2
    content_center = (
        top_mid[0] + inward_unit[0] * data_offset,
        top_mid[1] + inward_unit[1] * data_offset,
    )
    paste_along_axis(data_panel, content_center)


def suche_isbn_im_katalog(medium: dict) -> str:
    """Liefert den wahrscheinlichsten ISBN-Titel/Autor-Treffer aus Open Library."""
    title = (medium.get("titel") or "").strip()
    author = (medium.get("autor") or "").strip()
    if len(title) < 3:
        print("Open Library: Titel fuer ISBN-Suche zu kurz.")
        return ""

    params = {
        "title": title,
        "fields": "title,author_name,isbn_10,isbn_13",
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
    candidates_13 = {}
    candidates_10 = {}
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
                candidates_13[digits] = candidates_13.get(digits, 0) + 1

        for isbn in result.get("isbn_10", []):
            digits = re.sub(r"[^0-9Xx]", "", isbn).upper()
            if _isbn10_gueltig(digits):
                candidates_10[digits] = candidates_10.get(digits, 0) + 1

    # Rank candidates by how often they occur in title/author-matched records.
    print(
        f"Open Library fuer '{title}': {len(results)} Suchtreffer, "
        f"{len(candidates_13)} passende ISBN-13- und "
        f"{len(candidates_10)} passende ISBN-10-Kandidaten."
    )
    if candidates_13:
        identifier = sorted(candidates_13, key=lambda code: -candidates_13[code])[0]
        return identifier
    if candidates_10:
        identifier = sorted(candidates_10, key=lambda code: -candidates_10[code])[0]
        return identifier
    return ""


async def recherchiere_buch_parallel(medium: dict, image: Image.Image, semaphore: asyncio.Semaphore):
    """Recherchiert eine Buchausgabe separat; mehrere Buecher laufen parallel."""
    try:
        crop_base64 = await asyncio.to_thread(_medium_crop_base64, image, medium)
        async with semaphore:
            result = await asyncio.to_thread(_recherche_isbn_web, medium, crop_base64)
        medium.update(result)
    except Exception as exc:
        print(f"Webrecherche fehlgeschlagen fuer '{medium.get('titel', '')}': {exc}")

    raw_ident = (medium.get("isbn_ean") or "").strip().upper()
    compact_ident = re.sub(r"[^0-9X]", "", raw_ident)
    ident = (
        compact_ident
        if len(compact_ident) == 10 and _isbn10_gueltig(compact_ident)
        else re.sub(r"\D", "", raw_ident)
    )
    if not ident:
        try:
            ident = await asyncio.to_thread(suche_isbn_im_katalog, medium)
            if ident:
                medium["isbn_quelle"] = "Open Library"
        except Exception as exc:
            print(f"Open Library Fallback fehlgeschlagen fuer '{medium.get('titel', '')}': {exc}")
    medium["isbn_ean"] = ident
    medium.setdefault("auflage_nicht_eindeutig", False)
    return ident


async def query_bonavendi_many(identifiers: list[str]) -> list[tuple[float, str]]:
    """Fragt mehrere Codes mit einem Chromium-Prozess und begrenzter Parallelitaet ab."""
    results: list[tuple[float, str]] = [(0.0, "Kein Ankauf") for _ in identifiers]
    if not identifiers or not any(identifier.strip() for identifier in identifiers):
        return results

    browser = None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15"
            )
            semaphore = asyncio.Semaphore(4)

            async def query_one(index: int, identifier: str):
                id_clean = identifier.strip()
                if not id_clean:
                    return index, (0.0, "Kein Ankauf")

                async with semaphore:
                    page = await context.new_page()
                    api_data = []
                    api_response_received = asyncio.Event()

                    async def handle_response(response):
                        if "bonavendi" in response.url and response.status == 200:
                            content_type = response.headers.get("content-type", "")
                            if "json" in content_type:
                                try:
                                    body = await response.json()
                                    api_data.append(body)
                                    api_response_received.set()
                                except Exception:
                                    pass

                    try:
                        page.on("response", handle_response)
                        await page.goto(
                            f"https://www.bonavendi.de/verkaufen/?ean={id_clean}",
                            timeout=15000,
                            wait_until="domcontentloaded",
                        )
                        try:
                            await asyncio.wait_for(api_response_received.wait(), timeout=3)
                        except asyncio.TimeoutError:
                            pass

                        best_price = 0.0
                        best_vendor = "Kein Ankauf"
                        for data in api_data:
                            if isinstance(data, list):
                                offers = data
                            elif isinstance(data, dict):
                                offers = next(
                                    (data[key] for key in ("offers", "results", "data", "items") if data.get(key)),
                                    [],
                                )
                            else:
                                offers = []

                            for offer in offers:
                                if not isinstance(offer, dict):
                                    continue
                                for price_key in ("price", "buyPrice", "buy_price", "ankaufspreis", "value"):
                                    value = offer.get(price_key)
                                    if value:
                                        try:
                                            price = float(
                                                str(value).replace(",", ".").replace("\u20ac", "").strip()
                                            )
                                            if price > best_price:
                                                best_price = price
                                                best_vendor = (
                                                    offer.get("vendor") or offer.get("name") or
                                                    offer.get("shop") or "Ankaeufer"
                                                )
                                        except (TypeError, ValueError):
                                            pass

                        if best_price > 0:
                            print(f"Bonavendi {id_clean}: {best_price}\u20ac bei {best_vendor}")
                        return index, (best_price, best_vendor)
                    except Exception as exc:
                        print(f"Bonavendi Fehler fuer {id_clean}: {exc}")
                        return index, (0.0, "Kein Ankauf")
                    finally:
                        try:
                            await page.close()
                        except Exception:
                            pass

            try:
                pairs = await asyncio.gather(
                    *(query_one(index, identifier) for index, identifier in enumerate(identifiers))
                )
                for index, result in pairs:
                    results[index] = result
            finally:
                await browser.close()
                browser = None
    except Exception as exc:
        print(f"Bonavendi Browserfehler: {exc}")

    return results


async def kein_ankauf() -> tuple[float, str]:
    return 0.0, "Kein Ankauf"


@app.post("/")
async def scan_regal_root(file: UploadFile = File(...)):
    request_start = time.perf_counter()
    image_bytes = await file.read()
    detection_start = time.perf_counter()
    image, daten, draw = await asyncio.to_thread(verarbeite_das_bild, image_bytes)
    print(f"Bilderkennung ohne Websuche: {time.perf_counter() - detection_start:.2f}s.")

    medien = daten.get("medien", [])
    isbn_start = time.perf_counter()
    semaphore = asyncio.Semaphore(8)
    books = [medium for medium in medien if medium.get("typ") == "buch"]
    await asyncio.gather(*(
        recherchiere_buch_parallel(medium, image, semaphore)
        for medium in books
    ))
    print(
        f"ISBN-Recherchen fuer {len(books)} Buecher parallel abgeschlossen "
        f"in {time.perf_counter() - isbn_start:.2f}s."
    )

    for medium in medien:
        ident = medium.get("isbn_ean", "")
        if ident:
            print(
                f"ISBN/EAN fuer '{medium['titel']}' ({medium.get('ausgabe') or 'Ausgabe nicht erkannt'}): "
                f"{ident} | Quelle: {medium.get('isbn_quelle') or 'Open-Library-Fallback'}"
            )
        else:
            print(
                f"Keine ISBN gefunden fuer '{medium['titel']}'. Quelle: {medium.get('isbn_quelle') or '-'}"
            )
    print("GPT-Gesamterkennung:", json.dumps(daten, ensure_ascii=False))

    bonavendi_ids = [medium.get("isbn_ean", "") for medium in medien]

    if BONAVENDI_ENABLED:
        preise = await query_bonavendi_many(bonavendi_ids)
    else:
        print("Bonavendi-Abfrage deaktiviert; Preisermittlung wird uebersprungen.")
        preise = [(0.0, "Preissuche pausiert") for _ in medien]

    for i, medium in enumerate(medien):
        ident = medium.get("isbn_ean", "").strip()
        preis, anbieter = preise[i]

        if BONAVENDI_ENABLED:
            print(
                f"Medium: {medium['titel']} | Code: {ident or '-'} | "
                f"{preis}\u20ac bei {anbieter}"
            )
            price_text = f"Preis: {preis:.2f}\u20ac bei {anbieter}" if preis > 0 else "Kein Ankauf"
        else:
            print(f"Medium: {medium['titel']} | Code: {ident or '-'} | Preissuche pausiert")
            price_text = "Preis: pausiert"
        farbe_box = "#00E600"

        ecken = medium["ecken"]
        points = [
            (ecken[f"punkt_{punkt}"]["x"], ecken[f"punkt_{punkt}"]["y"])
            for punkt in range(1, 5)
        ]
        draw.line(points + [points[0]], fill=farbe_box, width=6)
        zeichne_buchtext(image, points, medium, price_text)

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    print(f"Gesamte Serververarbeitung: {time.perf_counter() - request_start:.2f}s.")
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
