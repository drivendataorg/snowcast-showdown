  After 03/30/22 [Aqua Safe Mode Alert](https://lpdaac.usgs.gov/news/aqua-safe-mode-alert/) as source of modis data use Terra spacecraft.
  Files loader_20220330-20220630.py
  Changed:
    line 90:             folder = "MYD10A1" + '/' + '{:0>2d}/{:0>2d}'.format(h,v) + '/' + f"{day:%Y%j}"
         to:             folder = "MOD10A1" + '/' + '{:0>2d}/{:0>2d}'.format(h,v) + '/' + f"{day:%Y%j}"
  Added line 98: filename = filename.replace("MOD10A1", "MYD10A1")

Loader work as dags for apache airflow.
