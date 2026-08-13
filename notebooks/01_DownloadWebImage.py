import urllib.request

url = "https://pytorch.tips/coffee"
fpath = "coffee.jpg"
# urllib.request.urlretrieve(url, fpath)

# claude suggestion: the standard script may be detected as a bot
# claude wants to send browser headers along

req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as response, open(fpath, "wb") as out_file:
    out_file.write(response.read())

# works
