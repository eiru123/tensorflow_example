def solution(dartResult):
    stack = []

    for c in dartResult:
        if c == 'S':
            continue
        elif c == 'D':
            num = stack.pop() ** 2
            stack.append(num)
        elif c == 'T':
            num = stack.pop() ** 3
            stack.append(num)
        elif c == '*':
            if len(stack) == 1:
                num = stack.pop() * 2
                stack.append(num)
            else:
                num2 = stack.pop() * 2
                num1 = stack.pop() * 2
                stack.append(num1)
                stack.append(num2)
        elif c == '#':
            num = stack.pop() * -1
            stack.append(num)
        else:
            num = int(c)
            stack.append(num)
    answer = sum(stack)

    return answer
str1 = 'FRANCH'
str2 = 'francho'
set1 = [c1 + c2 for c1, c2 in zip(str1[:-1].lower(), str1[1:].lower())
           if 'a' <= c1 <= 'z' and 'a' <= c2 <= 'z']
set2 = [c1 + c2 for c1, c2 in zip(str2[:-1].lower(), str2[1:].lower())
           if 'a' <= c1 <= 'z' and 'a' <= c2 <= 'z']
c = ["CCBDE", "AAADE", "AAABF", "CCBBF"]
print(c[1][2])
print(set1 in set2)
print(set1)
print(solution('1S2D*3T'))
