import os
import time
import requests
import pandas as pd
from tqdm import tqdm


CSV_PATH = "data/raw/test.csv"
OUTPUT_DIR = "data/images"

STYLE_ID = "mapbox/satellite-v9"
ZOOM = 18
IMG_WIDTH = 224
IMG_HEIGHT = 224

SLEEP_SEC = 0.1  # keeps you safely under rate limits


MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY")
if MAPBOX_API_KEY is None:
    raise RuntimeError(
        "MAPBOX_API_KEY not set. " "Run: export MAPBOX_API_KEY='your_token_here'"
    )


def fetch_satellite_image(lat: float, lon: float, out_path: str) -> bool:
    """
    Fetch a single satellite image centered at (lat, lon)
    using Mapbox Static Images API.
    """

    url = (
        f"https://api.mapbox.com/styles/v1/{STYLE_ID}/static/"
        f"{lon},{lat},{ZOOM}/{IMG_WIDTH}x{IMG_HEIGHT}@2x"
        f"?access_token={MAPBOX_API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10)
    except requests.RequestException as e:
        print(f"Request error @ ({lat}, {lon}): {e}")
        return False

    if response.status_code == 200:
        with open(out_path, "wb") as f:
            f.write(response.content)
        return True

    else:
        print(f"Failed [{response.status_code}] @ ({lat}, {lon})")
        return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    required_cols = {"id", "lat", "long"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")


    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Downloading satellite images",
        unit="img",
        miniters=10,     # update bar every ~10 iterations
    ):
        img_name = f"{int(row['id'])}.jpg"
        img_path = os.path.join(OUTPUT_DIR, img_name)

        # SAFETY: do not re-download
        if os.path.exists(img_path):
            continue

        fetch_satellite_image(
            lat=row["lat"],
            lon=row["long"],
            out_path=img_path
        )

        time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    main()
