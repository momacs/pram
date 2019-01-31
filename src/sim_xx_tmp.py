from enum import IntEnum

class A(IntEnum):
    a1 = 0
    a2 = 1

print(A)
print(A.a1)
print(A.a1.value)

B = IntEnum('B', 'b1 b2')

print(B)
print(B.b1)
print(B.b1.value)
