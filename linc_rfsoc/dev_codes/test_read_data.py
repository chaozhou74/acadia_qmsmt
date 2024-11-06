from acadia.data import DataManager

data_dir = r"/tmp/241104_143701"
dm = DataManager()
dm.load(data_dir)
print(dm.groups())