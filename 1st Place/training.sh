export MODISPATH=data/modis             #set path to local modis data cache

python3 src/main.py --maindir . --mode finalize --implicit 1 --relativer 0 --embedding 1 --individual 0
python3 src/main.py --maindir . --mode finalize --implicit 1 --relativer 0 --embedding 0 --individual 0
python3 src/main.py --maindir . --mode finalize --implicit 0 --relativer 0 --embedding 1 --individual 1
python3 src/main.py --maindir . --mode finalize --implicit 0 --relativer 0 --embedding 0 --individual 1
python3 src/main.py --maindir . --mode finalize --implicit 1 --relativer 0 --embedding 1 --individual 0 --modelsize B
python3 src/main.py --maindir . --mode finalize --implicit 1 --relativer 0 --embedding 0 --individual 0 --modelsize B
python3 src/main.py --maindir . --mode finalize --implicit 0 --relativer 0 --embedding 1 --individual 1 --modelsize B
python3 src/main.py --maindir . --mode finalize --implicit 0 --relativer 0 --embedding 0 --individual 1 --modelsize B