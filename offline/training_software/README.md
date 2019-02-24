## TODOS

```
[x] cp `stream_data.py`
[x] document what python packages `stream_data.py` requires
[] figure out how data is streamed in (probably will be like [1,2,...,8] for channels)
[] create a server to attach to `stream_data.py` (either inside or outside)
[] `server.[x]` should have an open socket to listen to `stream_data.py` input and output to the html file
[] create html file that has two buttons "left" and "right"
```

For installation:
```
pip3 install openbci-python
pip3 install serial
pip3 install pyserial
pip3 install yapsy
pip3 install pylsl
pip3 install python-osc
pip3 install requests
pip3 install xmltodict
```

Note there is also a requirements.txt file. 
You can install all requirements including the above packages with:

```
pip install -r requirements.txt
```
