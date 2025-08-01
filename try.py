from metrics.reward_functions import calculate_reward
from models.student import Student
from models.teacher import Teacher


if __name__ == '__main__':
    teacher = Teacher()
    problem = teacher()
    print(problem.model_dump())
    student = Student()
    solution = student(problem)
    print(solution.model_dump())
    reward = calculate_reward(solution.code, problem.test_cases)
    print(reward)

