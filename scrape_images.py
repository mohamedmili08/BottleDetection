from icrawler.builtin import BingImageCrawler
import os


def collect_data(keyword, folder_name, count=350):
    path = f'dataset/raw_images/{folder_name}'
    if not os.path.exists(path):
        os.makedirs(path)

    # Using Bing as it is often more 'friendly' to scrapers than Google
    crawler = BingImageCrawler(downloader_threads=4, storage={'root_dir': path})
    crawler.crawl(keyword=keyword, max_num=count)


# Execute for your 3 classes
collect_data("plastic water bottle closed cap", "OK")
collect_data("plastic bottle without cap", "No_Cap")
collect_data("crushed plastic bottle damaged", "Damaged")