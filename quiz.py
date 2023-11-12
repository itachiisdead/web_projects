'''
#quiz n1
def fact(n):
   if n < 0:
        print("number must be positive")
   elif n == 0 or n == 1:
        return 1
   else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

number = int(input("Enter a number: "))
result = fact(number)
print("The factorial of", number, "is", result)

'''






'''
dict = {
    'egypt': 'cairo',
    'england': 'london',
    'france': 'paris',
    'japan':'tokyo'
}

contry =input("Enter a contry: ")
if contry in dict:
    cap=dict[contry]
    print(contry,cap)

else:
    print('capital unkown') 

'''



class rect: 
    def __init__(self, len, wid):
        self.len = len
        self.wid = wid

    def area(self,len,wid):
        area=len*wid
        print('area is ',area )

    def perimeter(self,len,wid):
        per=(len*wid)+2
        print('perimeter is ', per)


a=rect(2,3)
a.area(2,3)
a.perimeter(2,3)