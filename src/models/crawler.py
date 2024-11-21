import asyncio
import concurrent.futures
import multiprocessing
import aiohttp.http_exceptions
import requests
import json
import os
import random
import time
import aiohttp


class DownloadStatus(object):
    def __init__(self, url: str, is_success: bool) -> None:
        self._url = url
        self._is_success = is_success

    @property
    def url(self) -> str:
        return self._url
    
    @property
    def is_success(self) -> bool:
        return self._is_success


class Crawler(object):
    '''
    # Gelbooru Crawler

    ---

    ## What is it for?

    It is a tool for crawling the images from gelbooru.
    The only thing you need to do is providing your api key and user id to it.
    It will automatically fetch the urls of the images and download them to your computer.

    ---

    ## Features
    + Simple interfaces.
    + Both sync and async crawing method.
    + Retry when failed.

    ---

    ## Limits

    To obey the policy of gelbooru, limits are set for this crawler.
    The parameters are defined below, chech the source code to know more about it.
    You can also modify the source code to exceed the limit (not recommended).

    ---

    ## Tests done

    ### Equipments
    + i9-14900k
    + m2 drive
    + 1Gbps ethernet
    
    ### Result
    + 1000 pictures in 37 secs
    + 5000 pictures in 173 secs

    #### Just so fast!
    '''
    __URL = "https://gelbooru.com/index.php?"
    __BASIC_PARAMS = {
        'page': 'dapi', 
        's': 'post', 
        'q': 'index', 
        'json': 1
    }

    __MAX_TASKS_LOWER_BOUND = 1
    __MAX_TASKS_UPPER_BOUND = multiprocessing.cpu_count()
    __MAX_TASKS_UPPER_BOUND_ASYNC = 100
    __MAX_TASKS_DEFAULT = __MAX_TASKS_UPPER_BOUND
    __MAX_TASKS_DEFAULT_ASYNC = __MAX_TASKS_UPPER_BOUND_ASYNC

    __PAGE_LOWER_BOUND = 1
    __PAGE_UPPER_BOUND = 200

    __PAGE_MIN_DEFAULT = __PAGE_LOWER_BOUND
    __PAGE_MAX_DEFAULT = __PAGE_UPPER_BOUND

    __PAGE_SIZE_LOWER_BOUND = 1
    __PAGE_SIZE_UPPER_BOUND = 100
    __PAGE_SIZE_DEFAULT = 10

    __CHUNK_SIZE_LOWER_BOUND = 1024
    __CHUNK_SIZE_UPPER_BOUND = 16384
    __CHUNK_SIZE_DEFAULT = 4096

    __WAIT_TIME_LOWER_BOUND = 1
    
    __WAIT_TIME_MIN_DEFAULT = 2
    __WAIT_TIME_MAX_DEFAULT = 3

    __IS_ASYNC_DEFAULT = False

    __DOWNLOAD_FOLDER_PATH_DEFAULT = "images"

    __RETRY_AFTER_SECS_DEFAULT = 5
    __MAX_RETRY_COUNTS_DEFAULT = 5

    __KEEP_IMAGES_ONLY_DEFAULT = False

    def __init__(self) -> None:
        self.__api_key = ""
        self.__user_id = ""

        self.__page_min = self.__PAGE_MIN_DEFAULT
        self.__page_max = self.__PAGE_MAX_DEFAULT
        self.__page_size = self.__PAGE_SIZE_DEFAULT

        self.__tags_str = ""

        self.__wait_time_min = self.__WAIT_TIME_MIN_DEFAULT
        self.__wait_time_max = self.__WAIT_TIME_MAX_DEFAULT

        self.__is_async = self.__IS_ASYNC_DEFAULT
        self.__max_tasks = self.__MAX_TASKS_DEFAULT if self.__is_async is False else self.__MAX_TASKS_DEFAULT_ASYNC

        self.__chunk_size = self.__CHUNK_SIZE_DEFAULT
        self.__download_folder_path = self.__get_download_folder_path(self.__DOWNLOAD_FOLDER_PATH_DEFAULT)

        self.__max_retry_counts = self.__MAX_RETRY_COUNTS_DEFAULT
        self.__retry_counts = 0
        self.__fetch_done = False

        self.__keep_images_only = self.__KEEP_IMAGES_ONLY_DEFAULT

    '''
    Define getters and setters below.
    Another comment will appear below to seperate the code into blocks.
    Do not remove this.
    '''

    @property
    def api_key(self) -> str:
        return self.__api_key
    
    @api_key.setter
    def api_key(self, value: str) -> None:
        self.__api_key = value

    @property
    def user_id(self) -> str:
        return self.__user_id
    
    @user_id.setter
    def user_id(self, value: str) -> None:
        self.__user_id = value

    @property
    def page_min(self) -> int:
        return self.__page_min
    
    @page_min.setter
    def page_min(self, value: int) -> None:
        if value < self.__PAGE_LOWER_BOUND:
            self.__page_min = self.__PAGE_LOWER_BOUND
        elif value > self.__PAGE_UPPER_BOUND:
            self.__page_min = self.__PAGE_UPPER_BOUND
        else:
            self.__page_min = value
    
    @property
    def page_max(self) -> int:
        return self.__page_max
    
    @page_max.setter
    def page_max(self, value: int) -> None:
        if value < self.__PAGE_LOWER_BOUND:
            self.__page_max = self.__PAGE_LOWER_BOUND
        elif value > self.__PAGE_UPPER_BOUND:
            self.__page_max = self.__PAGE_UPPER_BOUND
        else:
            self.__page_max = value

    @property
    def page_size(self) -> int:
        return self.__page_size
    
    @page_size.setter
    def page_size(self, value: int) -> None:
        if value < self.__PAGE_SIZE_LOWER_BOUND:
            self.__page_size = self.__PAGE_SIZE_LOWER_BOUND
        elif value > self.__PAGE_SIZE_UPPER_BOUND:
            self.__page_size = self.__PAGE_SIZE_UPPER_BOUND
        else:
            self.__page_size = value

    @property
    def tags_str(self) -> str:
        return self.__tags_str
    
    @tags_str.setter
    def tags_str(self, value: str) -> None:
        self.__tags_str = value
    
    @property
    def wait_time_min(self) -> float:
        return self.__wait_time_min
    
    @wait_time_min.setter
    def wait_time_min(self, value: float) -> None:
        if value < self.__WAIT_TIME_LOWER_BOUND:
            self.__wait_time_min = self.__WAIT_TIME_LOWER_BOUND
        else:
            self.__wait_time_min = value
    
    @property
    def wait_time_max(self) -> float:
        return self.__wait_time_max
    
    @wait_time_max.setter
    def wait_time_max(self, value: float) -> None:
        if value < self.__WAIT_TIME_LOWER_BOUND:
            self.__wait_time_max = self.__WAIT_TIME_LOWER_BOUND
        else:
            self.__wait_time_max = value

    @property
    def max_tasks(self) -> int:
        return self.__max_tasks
    
    @max_tasks.setter
    def max_tasks(self, value: int) -> None:
        if value < self.__MAX_TASKS_LOWER_BOUND:
            self.__max_tasks  = self.__MAX_TASKS_LOWER_BOUND
            return None
        
        if self.__is_async is True:
            if value > self.__MAX_TASKS_UPPER_BOUND_ASYNC:
                self.__max_tasks = self.__MAX_TASKS_UPPER_BOUND_ASYNC
            else:
                self.__max_tasks = value
        else:
            if value > self.__MAX_TASKS_UPPER_BOUND:
                self.__max_tasks = self.__MAX_TASKS_UPPER_BOUND
            else:
                self.__max_tasks = value

    @property
    def chunk_size(self) -> int:
        return self.__chunk_size
    
    @chunk_size.setter
    def chunk_size(self, value: int) -> int:
        if value < self.__CHUNK_SIZE_LOWER_BOUND:
            self.__chunk_size = self.__CHUNK_SIZE_LOWER_BOUND
        elif value > self.__CHUNK_SIZE_UPPER_BOUND:
            self.__chunk_size = self.__CHUNK_SIZE_UPPER_BOUND
        else:
            self.__chunk_size = value

    @property
    def __processed_tags_str(self) -> str:
        return "+".join(self.__tags_str.split())
    
    @property
    def __auth_params(self) -> dict[str, str]:
        return {
            'api_key': self.__api_key, 
            'user_id': self.__user_id
        }
    
    @property
    def is_async(self) -> bool:
        return self.__is_async
    
    @is_async.setter
    def is_async(self, value: bool) -> None:
        self.__is_async = value

    @property
    def download_folder_path(self) -> str:
        return self.__get_download_folder_path(self.__download_folder_path)
    
    @download_folder_path.setter
    def download_folder_path(self, value: str) -> None:
        self.__download_folder_path = os.path.abspath(value)
    
    @property
    def __pages(self) -> list[int]:
        return list(range(self.__page_min, self.__page_max + 1))
    
    @property
    def max_threads_can_use(self) -> int:
        return self.__MAX_TASKS_UPPER_BOUND
    
    @property
    def download_folder_exists(self) -> bool:
        return True if os.path.exists(self.download_folder_path) else False
    
    @property
    def keep_images_only(self) -> bool:
        return self.__keep_images_only
    
    @keep_images_only.setter
    def keep_images_only(self, value: bool) -> None:
        self.__keep_images_only = value
    
    '''
    Define methods below.
    Another comment will appear below to seperate the code into blocks.
    Do not remove this.
    '''

    def summary(self) -> None:
        print(self.__dict__)
    
    def __get_search_params(self, page: int) -> dict[str, str]:
        return {
            'limit': self.__page_size, 
            'pid': page, 
            'tags': self.__processed_tags_str
        }
    
    def __get_params_str(self, params: dict) -> str:
        return "&".join(("%s=%s" % (key, value) for key, value in params.items()))
    
    def __get_download_folder_path(self, value: str) -> str:
        return os.path.abspath(value)
    
    def __reset_fetch_statuses(self) -> None:
        self.__retry_counts = 0
        self.__fetch_done = False
    
    def __correct_parameters(self):
        '''
        Always be called before fetching urls.
        This method swaps page_min and page_max if the forward one is bigger than the backward one.
        Same logic for wait_time_min and wait_time_max.
        '''
        if self.__page_min > self.__page_max:
            self.__page_min, self.__page_max = self.__page_max, self.__page_min

        if self.__wait_time_min > self.__wait_time_max:
            self.__wait_time_min, self.__wait_time_max = self.__wait_time_max, self.__wait_time_min

    def __create_download_folder(self):
        os.mkdir(self.download_folder_path)
    
    def __get_image_url_by_page(self, page: int) -> list[str]:
        params = self.__BASIC_PARAMS | self.__auth_params | self.__get_search_params(page=page)
        params_str = self.__get_params_str(params=params)

        url = self.__URL + params_str

        print(f"Fetching image urls. Page: {page}")
        response = requests.post(url=url)
        print(f"Fetch done. Page: {page}")

        response_content_json = json.loads(response.content)
        image_infos = response_content_json.get("post")

        if not image_infos:
            image_infos = []

        image_urls = []

        for image_info in image_infos:
            image_url = image_info.get("file_url")
            if not image_url:
                continue
            image_urls.append(image_url)

        time.sleep(random.random() * (self.__wait_time_max - self.__wait_time_min) + self.__wait_time_min)
        
        return image_urls
    
    async def __get_image_url_by_page_async(self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, page: int) -> list[str]:
        params = self.__BASIC_PARAMS | self.__auth_params | self.__get_search_params(page=page)
        params_str = self.__get_params_str(params=params)

        url = self.__URL + params_str

        async with semaphore:
            print(f"Fetching image urls. Page: {page}")
            response = await session.post(url=url)
            print(f"Fetch done. Page: {page}")

        response_content_json = await response.json()

        image_infos = response_content_json.get("post")

        if not image_infos:
            image_infos = []

        image_urls = []

        for image_info in image_infos:
            image_url = image_info.get("file_url")
            if not image_url:
                continue
            image_urls.append(image_url)
        
        return image_urls
    
    def get_image_urls(self) -> list[str]:
        self.__correct_parameters()

        image_urls = []

        with concurrent.futures.ThreadPoolExecutor(self.__max_tasks) as executor:
            image_url_lists = executor.map(self.__get_image_url_by_page, self.__pages)
            for image_url_list in image_url_lists:
                image_urls.extend(image_url_list)
        
        return image_urls
    
    async def get_image_urls_async(self) -> list[str]:

        self.__correct_parameters()

        session = aiohttp.ClientSession()
        semaphore = asyncio.Semaphore(self.__max_tasks)

        url_lists = await asyncio.gather(*[self.__get_image_url_by_page_async(session=session, semaphore=semaphore, page=page) for page in self.__pages])
        await session.close()

        urls = []

        for url_list in url_lists:
            urls.extend(url_list)

        return urls
    
    def __download_image_by_url(self, url: str) -> None:
        if not self.download_folder_exists:
            self.__create_download_folder()

        filename = url.split("/")[-1]
        if not filename.endswith(".jpg") and not filename.endswith(".png"):
            return None

        print(f"Downloading image. Url: {url}")

        path = f"{self.download_folder_path}/{filename}"
        with requests.get(url=url, stream=True) as response:
            with open(file=path, mode='wb') as file:
                for content in response.iter_content(chunk_size=self.__chunk_size):
                    file.write(content)

        print(f"Download done. Image path: {path}")

        time.sleep(random.random() * (self.__wait_time_max - self.__wait_time_min) + self.__wait_time_min)

    async def __download_image_by_url_async(self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, url: str) -> DownloadStatus:
        '''
        Return True if download is successful else False.
        '''

        filename = url.split("/")[-1]
        if not filename.endswith(".jpg") and not filename.endswith(".png"):
            return DownloadStatus(url=url, is_success=True)

        async with semaphore:
            try:
                async with session.get(url=url) as response:
                    print(f"Downloading image. Url: {url}")
                    path = f"{self.download_folder_path}/{filename}"

                    with open(file=path, mode='wb') as file:
                        async for line in response.content.iter_chunked(self.__chunk_size):
                            file.write(line)

                    print(f"Download done. Image filename: {path}")
                
                await asyncio.sleep(random.random() * (self.__wait_time_max - self.__wait_time_min) + self.__wait_time_min)
                return DownloadStatus(url=url, is_success=True)
            except Exception:
                print(f"Download failed. Image url: {url}")
                return DownloadStatus(url=url, is_success=False)
            
        return DownloadStatus(url=url, is_success=True)
                   
    def download_images(self, urls: list[str]) -> None:
        self.__correct_parameters()

        with concurrent.futures.ThreadPoolExecutor(self.__max_tasks) as executor:
            executor.map(self.__download_image_by_url, urls)

    async def download_images_async(self, urls: list[str]) -> None:
        '''
        Like download_images(), this method will download all the images.
        However, it is done asynchronously here.

        Retry is implemented now, but using recursive.
        Create another method to handle the retry logic might be better.
        '''
        if self.__fetch_done is True:
            self.__reset_fetch_statuses()

        if not self.download_folder_exists:
            self.__create_download_folder()

        self.__correct_parameters()

        session = aiohttp.ClientSession()
        semaphore = asyncio.Semaphore(self.__max_tasks)
        #Download images asynchronously.
        download_status_list = await asyncio.gather(*[self.__download_image_by_url_async(session=session, semaphore=semaphore, url=url) for url in urls])

        #Always close the session after downloading the images.
        await session.close()

        #Retry check
        urls_required_retry = [download_status.url for download_status in download_status_list if download_status.is_success is False]
        urls_required_retry_len = len(urls_required_retry)

        if urls_required_retry_len > 0:
            if self.__retry_counts >= self.__max_retry_counts:
                print("Max retry times arrived.")
                return None

            retry_after_secs = self.__RETRY_AFTER_SECS_DEFAULT * (2 ** self.__retry_counts)
            self.__retry_counts = self.__retry_counts + 1

            print(f"{urls_required_retry_len} jobs failed. Retry after {retry_after_secs} secs.")

            await asyncio.sleep(retry_after_secs)
            await self.download_images_async(urls=urls_required_retry)
        else:
            self.__fetch_done = True
            print("All jobs are done!")
