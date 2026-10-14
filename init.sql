-- 关键：告诉 MySQL 后面的字节按 utf8mb4 解析
-- 否则客户端默认 latin1，中文会被双重编码，存进去就是乱码
SET NAMES utf8mb4;

-- 建表（与 models/ 里的 SQLAlchemy 模型保持一致）
CREATE TABLE IF NOT EXISTS students (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) NOT NULL,
    age INT NOT NULL,
    major VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL
);

-- 初始示例数据
INSERT INTO students (name, age, major) VALUES
('张三', 20, '人工智能'),
('李四', 21, '计算机科学与技术'),
('王五', 22, '软件工程');
