import dsu


class DSU(object):
    __dsu: dsu.DSU

    def __init__(self, size: int):
        self.__dsu = dsu.DSU(size)

    def find(self, item: int):
        return self.__dsu.find(item)

    def unite(self, first: int, second: int):
        self.__dsu.unite(first, second)

    def is_singleton(self):
        return self.__dsu.is_singleton()
