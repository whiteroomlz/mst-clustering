import types
import base64
import marshal
import functools

from typing import Union
from multiprocessing.sharedctypes import RawArray, RawValue, Array, Value
from concurrent.futures import ProcessPoolExecutor

shared_memory: dict[str, Array or Value or RawArray or RawValue]


def pool_init(shared_memory_dict: dict):
    global shared_memory
    shared_memory = shared_memory_dict.copy()


def call_submittable_function(function_info):
    args = function_info["args"]
    kwargs = function_info["kwargs"]
    encoded_function = function_info["encoded_function"]

    code = marshal.loads(base64.b64decode(encoded_function))
    func = types.FunctionType(code, globals(), "wrapped_func")
    return func(*args, **kwargs)


def submittable(function):
    @functools.wraps(function)
    def _wrapper(*args, **kwargs):
        encoded_function = base64.b64encode(marshal.dumps(function.__code__, 0))
        function_info = {"encoded_function": encoded_function, "args": args, "kwargs": kwargs}
        return function_info

    return _wrapper


class SharedMemoryPool(ProcessPoolExecutor):
    functions = dict()

    def __init__(self, max_workers: int, shared_memory_dict: dict[str, Array or Value or RawArray or RawValue]):
        super().__init__(max_workers=max_workers, initializer=pool_init, initargs=(shared_memory_dict,))

    def submit(self, function, *args, **kwargs):
        return super(SharedMemoryPool, self).submit(call_submittable_function, function(*args, **kwargs))

    @staticmethod
    def get_shared_memory() -> dict[str, Union[Array, Value, RawArray, RawValue]]:
        global shared_memory
        return shared_memory.copy()
