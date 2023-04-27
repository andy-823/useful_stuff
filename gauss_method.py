import numpy as np

# чисто служебная штука
def matrix_string(A:np.array, accuracy:int=3):
    output = ''
    n = len(A)
    for i in range(0, len(A)):
        # то, что в матрице слева
        for j in range(0, n):
            output += '\t%.*f '%(accuracy, A[i][j])
        # то, что в матрице y
        output += '\t| %.*f \t| '%(accuracy, A[i][n])
        # то, что в матрице справа
        for j in range(n + 1, len(A[i]) - 1):
            output += '%.*f \t'%(accuracy, A[i][j])
        output += '%.*f'%(accuracy, A[i][len(A[i]) - 1])
        output += '\n' # строка закончилась
    return output           

def solve(A:np.array,
          y:np.array, 
          show_output:bool=False,
          accuracy:int=5):
    assert len(A) == len(A[0]), "A не квадратная"
    assert len(A) == len(y), "Длины A и y различаются"

    output = ''
    n = len(A)
    A_list = []
    # создаем единичную матрицу
    E = np.array(np.zeros((n, n), dtype=float))
    for i in range(0,n): 
        E[i][i] = 1
    # создаем одну длинную табличку
    # она имеет вид (A | y | E)
    for i in range(0, n):
        A_list_i = A[i].tolist()
        A_list_i.append(y[i])
        A_list_i.extend(E[i].tolist())

        A_list.append(A_list_i)
    
    # A_extended, но название слишком длинное
    A_ext = np.array(A_list, dtype=float)
    if show_output:
        output += 'Исходная матрица:\n'
        output += matrix_string(A_ext, accuracy)
    
    count_swaps = 0
    for j in range(0, n - 1):
        if show_output: output += '\n'

        # поиск максимума по модулю на j-й строке
        max_val = abs(A_ext[j][j])
        max_num = j
        for i in range(j + 1, n):
            if abs(max_val) < abs(A_ext[i][j]):
                max_val = max(max_val, abs(A_ext[i][j]))
                max_num = i
        # свапаем с максимумом по модулю
        # количество свапов нужно для знака определителя
        if max_num != j:
            if show_output:
                output += 'A[%s] <-> A[%s]\n'%(j + 1, max_num + 1)
            # не думаю что это оптимально
            # но вряд ли это будет использоваться для больших данных
            A_ext[[j, max_num]] = A_ext[[max_num, j]]
            count_swaps += 1
            # if show_output: output += matrix_string(A_ext, accuracy)
        
        # в общем-то вычитание
        # мы дошли до j-го символа, если идти слева направо
        # то есть вычитаем j-ю строку домноженную на какой то коэффициент
        # из строчек пониже (номер у них побольше)
        for i in range(j + 1, n):
            assert abs(A_ext[j][j]) > 1e-10, "Матрица должна быть невырожденная"
            coefficient = A_ext[i][j] / A_ext[j][j]
            if show_output:
                output += 'A[%s] -= %s * A[%s]\n'%(i + 1, round(coefficient, accuracy), j + 1)
            A_ext[i] = A_ext[i] - coefficient * A_ext[j]
        if show_output: output += matrix_string(A_ext, accuracy)

    # определитель
    det = 1
    for i in range(0, n):
        det *= A_ext[i][i]
    det *= (-1)**count_swaps

    # теперь приводим к диагональному виду
    i = n - 1
    while i > 0:
        if show_output: output += '\n'
        for j in range(0, i):
            coefficient = A_ext[j][i] / A_ext[i][i]
            A_ext[j] -= coefficient * A_ext[i]
            if show_output:
                output += 'A[%s] -= %s * A[%s]\n'%(j + 1, round(coefficient, accuracy), i + 1)
        if show_output: output += matrix_string(A_ext, accuracy)
        i -= 1

    # теперь вид диагональный
    # теперь приводим матрицу слева 1-ному виду
    if show_output: output += '\n'
    for i in range(0, n):
        if show_output:
            output += 'A[%s] /= %s\n'%(i + 1, round(A_ext[i][i], accuracy))
        A_ext[i] /= A_ext[i][i]
    if show_output: output += matrix_string(A_ext, accuracy)

    # вектор ответ
    x = np.zeros(y.shape, dtype=float)
    for i in range(0, n):
        x[i] = A_ext[i][n]
    
    # обратная матрица
    inverted = []
    for i in range(0, n):
        inverted_i = []
        for j in range(n + 1, len(A_ext[i])):
            inverted_i.append(A_ext[i][j])
        inverted.append(inverted_i)
    inverted = np.array(inverted, dtype=float)

    if show_output:
        print(output)
    
    return x, det, inverted


A = np.array([
    [1, 4, -9, 7],
    [2, -2, -2, 3],
    [-1, 3, -9, -1],
    [-5, 2, 2, 1]
], dtype=float)
y = np.array([-67, -57, -26, 52], dtype = float)

x, det, inv = solve(A=A, y=y, show_output=True, accuracy=3)
print(x)
print(A @ x)
print(y)
print()
print(det)
print()
print(inv)
print(A @ inv)