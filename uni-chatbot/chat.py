import sys
import requests

question = sys.argv[1]

response = requests.post(
    "http://127.0.0.1:5000/ask",
    json={"question": question}
)

data = response.json()

print("\nANSWER:\n")
print(data["answer"])

if data.get("sources"):
    print("\nSOURCES:")
    for s in data["sources"]:
        print("-", s)