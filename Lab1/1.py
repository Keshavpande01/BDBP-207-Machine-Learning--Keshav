
A = [
    [1, 2, 3],
    [4, 5, 6]
]
rows = len(A)
cols = len(A[0])

AT = []
for i in range(cols):
    row = []
    for j in range(rows):
        row.append(A[j][i])
    AT.append(row)

ATA = []

for i in range(len(AT)):          # rows of AT
    row = []
    for j in range(len(A[0])):    # columns of A
        s = 0
        for k in range(len(A)):   # columns of AT / rows of A
            s += AT[i][k] * A[k][j]
        row.append(s)
    ATA.append(row)
for r in ATA:
    print(r)
