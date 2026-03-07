from bing_image_downloader import downloader

print("Downloading TEST dataset...")

downloader.download(
    "indian dal curry",
    limit=10,
    output_dir="food_dataset/test",
    adult_filter_off=True,
    force_replace=False
)

downloader.download(
    "cooked white rice",
    limit=10,
    output_dir="food_dataset/test",
    adult_filter_off=True,
    force_replace=False
)

downloader.download(
    "indian roti chapati",
    limit=10,
    output_dir="food_dataset/test",
    adult_filter_off=True,
    force_replace=False
)

downloader.download(
    "indian food dish",
    limit=10,
    output_dir="food_dataset/test",
    adult_filter_off=True,
    force_replace=False
)

print("Test dataset ready!")