import pathlib, sqlite3, pprint

db = pathlib.Path("dataset/database/博金杯比赛数据.db")
con = sqlite3.connect(db)
cur = con.execute("select name from sqlite_master where type='table'")
pprint.pp(cur.fetchall())