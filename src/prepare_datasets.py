from models import Crawler

import os
import asyncio
import random
import shutil

async def main():
    crawler = Crawler()
    crawler.is_async = True
    crawler.keep_images_only = True

    crawler.api_key = "0c893db2cb9cacfb7ef5185a0d936ad047161c48dd3bcaffa6e5557e4807ef48"
    crawler.user_id = "1392188"
    crawler.page_min = 1
    crawler.page_max = 20
    crawler.page_size = 100
    crawler.max_tasks = 100
    crawler.wait_time_min = 2
    crawler.wait_time_max = 3

    labels: list[str] = []
    with open("datasets/labels.txt", mode="r", encoding="utf8") as file:
        while True:
            line = file.readline()
            if line == "":
                break
            line = line.strip()
            if line == "":
                continue
            labels.append(line)

    datasets_root_path = os.path.abspath("datasets")
    if not os.path.exists(datasets_root_path):
        os.mkdir(datasets_root_path)

    datasets_raw_path = os.path.join(datasets_root_path, "raw")
    datasets_train_path = os.path.join(datasets_root_path, "train")
    datasets_validation_path = os.path.join(datasets_root_path, "validation")
    datasets_test_path = os.path.join(datasets_root_path, "test")

    if not os.path.exists(datasets_raw_path):
        os.mkdir(datasets_raw_path)
    if not os.path.exists(datasets_train_path):
        os.mkdir(datasets_train_path)
    if not os.path.exists(datasets_validation_path):
        os.mkdir(datasets_validation_path)
    if not os.path.exists(datasets_test_path):
        os.mkdir(datasets_test_path)

    labels = [label.strip() for label in labels]
    
    for label in labels:
        crawler.tags_str = label
        crawler.download_folder_path = os.path.join(datasets_raw_path, label).strip()

        datasets_train_label_path = os.path.join(datasets_train_path, label).strip()
        datasets_validation_label_path = os.path.join(datasets_validation_path, label).strip()
        datasets_test_label_path = os.path.join(datasets_test_path, label).strip()

        if not os.path.exists(crawler.download_folder_path):
            urls = await crawler.get_image_urls_async()
            urls = random.sample(urls, k=300)
            await crawler.download_images_async(urls=urls)

        paths = os.listdir(crawler.download_folder_path)
        paths_len = len(paths)
        paths_train = random.sample(paths, k=int(paths_len * 0.6))
        paths_other = [path for path in paths if path not in paths_train]
        paths_other_len = len(paths_other)
        paths_validation = random.sample(paths_other, k=int(paths_other_len * 0.5))
        paths_test = [path for path in paths_other if path not in paths_validation]

        if not os.path.exists(datasets_train_label_path):
            os.mkdir(datasets_train_label_path)
        if not os.path.exists(datasets_validation_label_path):
            os.mkdir(datasets_validation_label_path)
        if not os.path.exists(datasets_test_label_path):
            os.mkdir(datasets_test_label_path)

        for path in paths_train:
            shutil.copyfile(os.path.join(crawler.download_folder_path, path).strip(), os.path.join(datasets_train_label_path, path).strip())
        for path in paths_validation:
            shutil.copyfile(os.path.join(crawler.download_folder_path, path).strip(), os.path.join(datasets_validation_label_path, path).strip())
        for path in paths_test:
            shutil.copyfile(os.path.join(crawler.download_folder_path, path).strip(), os.path.join(datasets_test_label_path, path).strip())

if __name__ == "__main__":
    asyncio.run(main())