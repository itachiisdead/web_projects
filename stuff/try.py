# Create a dictionary to store student information
student = {
    'name': 'John Smith',
    'age': 20,
    'major': 'Computer Science',
    'gpa': 3.8
}

# Accessing values in the dictionary
print("Name:", student['name'])
print("Age:", student['age'])
print("Major:", student['major'])
print("GPA:", student['gpa'])

# Modifying values in the dictionary
student['age'] = 21
student['gpa'] = 3.9

# Adding a new key-value pair to the dictionary
student['university'] = 'ABC University'

# Deleting a key-value pair from the dictionary
del student['major']

# Iterating over the dictionary
for key, value in student.items():
    print(key + ":", value)













# Define a class named "Person"
class Person:
    # Constructor method to initialize the object
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # Method to greet the person
    def greet(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

    # Method to update the age of the person
    def update_age(self, new_age):
        self.age = new_age

# Create an instance of the Person class
person1 = Person("John", 25)

# Accessing attributes of the person object
print("Name:", person1.name)
print("Age:", person1.age)

# Calling methods on the person object
person1.greet()

# Updating the age of the person
person1.update_age(26)
print("Updated Age:", person1.age)


















def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    elif n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

# Test the factorial function
number = int(input("Enter a number: "))
result = factorial(number)
print("The factorial of", number, "is", result)