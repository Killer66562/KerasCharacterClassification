import tkinter as tk
import math

from .crawler import Crawler


class CrawlerGUI(object):
    def __init__(self) -> None:
        self.__crawler = Crawler()
        
        root = root = tk.Tk(sync=True)
        root.title("GelbooruCrawler")
        root.resizable(False, False)

        window_width = root.winfo_screenwidth()
        window_height = root.winfo_screenheight()

        width = 800
        height = 600

        left = (window_width - width) // 2
        top = (window_height - height) // 2

        root.geometry(f"{width}x{height}+{left}+{top}")

        auth_frame = tk.Frame(root)
        auth_frame.pack(side="top", fill="x")

        tags_frame = tk.LabelFrame(root, text="tags")
        tags_frame.pack(side="top", fill="x")

        page_frame = tk.Frame(root)
        page_frame.pack(side="top", fill="x")

        optimize_frame = tk.Frame(root)
        optimize_frame.pack(side="top", fill="x")

        def fetch():
            self.__crawler.api_key = api_key_var.get()
            self.__crawler.user_id = user_id_var.get()
            self.__crawler.tags_str = tags_var.get()

            try:
                fetch_button["state"] = tk.DISABLED
                urls = self.__crawler.get_image_urls()
                self.__crawler.download_images(urls)
            except:
                print("Error!")
            finally:
                fetch_button["state"] = tk.NORMAL

        fetch_button = tk.Button(root, text="抓取資料", command=fetch)
        fetch_button.pack(side="top", fill="x")

        api_key_frame = tk.LabelFrame(auth_frame, text="api key")
        api_key_frame.pack(side="left", fill="x", expand=1)

        user_id_frame = tk.LabelFrame(auth_frame, text="user id")
        user_id_frame.pack(side="right", fill="x", expand=1)

        page_min_frame = tk.LabelFrame(page_frame, text="page min")
        page_min_frame.pack(side="left", fill="x", expand=1)

        page_max_frame = tk.LabelFrame(page_frame, text="page max")
        page_max_frame.pack(side="left", fill="x", expand=1)

        page_size_frame = tk.LabelFrame(page_frame, text="page size")
        page_size_frame.pack(side="left", fill="x", expand=1)

        wait_time_min_frame = tk.LabelFrame(optimize_frame, text="wait time min")
        wait_time_min_frame.pack(side="left", fill="x", expand=1)

        wait_time_max_frame = tk.LabelFrame(optimize_frame, text="wait time max")
        wait_time_max_frame.pack(side="left", fill="x", expand=1)

        max_tasks_frame = tk.LabelFrame(optimize_frame, text="max tasks")
        max_tasks_frame.pack(side="left", fill="x", expand=1)

        api_key_var = tk.StringVar(value="")
        api_key_entry = tk.Entry(api_key_frame, textvariable=api_key_var)
        api_key_entry.pack(side="bottom", fill="x")

        user_id_var = tk.StringVar(value="")
        user_id_entry = tk.Entry(user_id_frame, textvariable=user_id_var)
        user_id_entry.pack(side="bottom", fill="x")

        tags_var = tk.StringVar(value="")
        tags_entry = tk.Entry(tags_frame, textvariable=tags_var)
        tags_entry.pack(side="bottom", fill="x")

        def on_page_min_spinbox_change(*args, **kwargs):
            self.__crawler.page_min = page_min_var.get()
            page_min_var.set(self.__crawler.page_min)
            root.focus()

        def on_page_max_spinbox_change(*args, **kwargs):
            self.__crawler.page_max = page_max_var.get()
            page_max_var.set(self.__crawler.page_max)
            root.focus()

        def on_page_size_spinbox_change(*args, **kwargs):
            self.__crawler.page_size = page_size_var.get()
            page_size_var.set(self.__crawler.page_size)
            root.focus()

        page_min_var = tk.IntVar(value=self.__crawler.page_min)
        page_min_spinbox = tk.Spinbox(page_min_frame, textvariable=page_min_var, from_=1, to=20000, increment=1, command=on_page_min_spinbox_change)
        page_min_spinbox.pack(side="bottom", fill="x")
        page_min_spinbox.bind("<Return>", on_page_min_spinbox_change)

        page_max_var = tk.IntVar(value=self.__crawler.page_max)
        page_max_spinbox = tk.Spinbox(page_max_frame, textvariable=page_max_var, from_=1, to=20000, increment=1, command=on_page_max_spinbox_change)
        page_max_spinbox.pack(side="bottom", fill="x")
        page_max_spinbox.bind("<Return>", on_page_max_spinbox_change)

        page_size_var = tk.IntVar(value=self.__crawler.page_size)
        page_size_spinbox = tk.Spinbox(page_size_frame, textvariable=page_size_var, from_=1, to=20000, increment=1, command=on_page_size_spinbox_change)
        page_size_spinbox.pack(side="bottom", fill="x")
        page_size_spinbox.bind("<Return>", on_page_size_spinbox_change)

        def on_wait_time_min_spinbox_change(*args, **kwargs):
            self.__crawler.wait_time_min = wait_time_min_var.get()
            wait_time_min_var.set(self.__crawler.wait_time_min)
            root.focus()

        def on_wait_time_max_spinbox_change(*args, **kwargs):
            self.__crawler.wait_time_max = wait_time_max_var.get()
            wait_time_max_var.set(self.__crawler.wait_time_max)
            root.focus()

        def on_max_tasks_spinbox_change(*args, **kwargs):
            self.__crawler.page_size = max_tasks_var.get()
            max_tasks_var.set(self.__crawler.max_tasks)
            root.focus()

        wait_time_min_var = tk.DoubleVar(value=self.__crawler.wait_time_min)
        wait_time_min_spinbox = tk.Spinbox(wait_time_min_frame, textvariable=wait_time_min_var, from_=1, to=5, increment=0.1, command=on_wait_time_min_spinbox_change)
        wait_time_min_spinbox.pack(side="bottom", fill="x")
        wait_time_min_spinbox.bind("<Return>", on_wait_time_min_spinbox_change)

        wait_time_max_var = tk.DoubleVar(value=self.__crawler.wait_time_max)
        wait_time_max_spinbox = tk.Spinbox(wait_time_max_frame, textvariable=wait_time_max_var, from_=2, to=10, increment=0.1, command=on_wait_time_max_spinbox_change)
        wait_time_max_spinbox.pack(side="bottom", fill="x")
        wait_time_max_spinbox.bind("<Return>", on_wait_time_max_spinbox_change)

        max_tasks_var = tk.IntVar(value=self.__crawler.max_tasks)
        max_tasks_spinbox = tk.Spinbox(max_tasks_frame, textvariable=max_tasks_var, from_=1, to=self.__crawler.max_threads_can_use, increment=1, command=on_max_tasks_spinbox_change)
        max_tasks_spinbox.pack(side="bottom", fill="x")
        max_tasks_spinbox.bind("<Return>", on_max_tasks_spinbox_change)

        self.__root = root

    def show(self) -> None:
        self.__root.mainloop()

    def exit(self) -> None:
        quit()
