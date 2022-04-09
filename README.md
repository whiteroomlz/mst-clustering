# Spanning tree clustering

Для инсталляции необходимо воспользоваться командой:

```bash
    pip install git+https://github.com/Whiteroomlz/Spanning-tree-clustering.git
```

Класс `SpanningTreeClustering` использует
модуль [multiprocessing](https://docs.python.org/3/library/multiprocessing.html), из-за чего требуется наличие точки
входа:

```python
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    ...
```
