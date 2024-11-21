from abc import ABC, abstractmethod


class Reader(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def read(self, path: str) -> list[str]:
        pass


class TextParamsReader(Reader):
    def __init__(self):
        super().__init__()

    def read(self, path: str) -> list[str]:
        args_list = []
        with open(file=path, mode="r", encoding="utf8") as file:
            while True:
                args = file.readline()
                if args == "":
                    break
                args = args.strip()
                if args == "":
                    continue
                args_list.append(args)
        return args_list