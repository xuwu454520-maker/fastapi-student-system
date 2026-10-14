CREATE DATABASE IF NOT EXISTS school
CHARACTER SET utf8mb4;

USE school;

CREATE TABLE IF NOT EXISTS students (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) NOT NULL,
    age INT NOT NULL,
    major VARCHAR(100) NOT NULL
);

INSERT INTO students (name, age, major)
VALUES
('张三', 20, '人工智能'),
('李四', 21, '计算机科学与技术');

SELECT * FROM students;