# Spanning tree clustering

Для инсталляции необходимо воспользоваться командой:

    C:\Users\borod>pip install git+https://github.com/Whiteroomlz/Spanning-tree-clustering.git

Класс `SpanningTreeClustering` использует
модуль [multiprocessing](https://docs.python.org/3/library/multiprocessing.html), из-за чего требуется использование
точки входа:

```python
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    ...
```
