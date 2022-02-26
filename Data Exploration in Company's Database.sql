CREATE DATABASE MYEXP;
USE MYEXP;
SELECT *
FROM PYTHON;

SELECT *
FROM FOLLOWUPFINAL;

SELECT PY.NAME,FF.NAME
FROM PYTHON PY
RIGHT JOIN FOLLOWUPFINAL FF
ON PY.NAME=FF.NAME;

SELECT *
FROM SQL_FILE;

SELECT FF.name,SF.name
FROM followupfinal FF
right outer join SQL_FILE SF
ON FF.name=SF.name;

CREATE DATABASE SQLPRACTICE;

USE SQLPRACTICE;

create table employee(empid int primary key,empname varchar(100), department varchar(50),
contactno bigint, emaildid varchar(100), empheadid int);

SELECT *
FROM EMPLOYEE;

INSERT INTO EMPLOYEE
VALUES
(101,"IshA","E-101",1234567890,"isha@gmail.com",105),
(102,"Priya","E-104",1234567890,"priya@yahoo.com",103),
(103,"Neha","E-101",1234567890,"neha@gmail.com",101),
(104,"Rahul","E-102",1234567890,"rahul@yahoo.com",105),
(105,"Abhishek","E-101",1234567890,"abhishek@gmail.com",102);

create table empdept(deptid varchar(50) primary key,deptname varchar(100), dept_off varchar(100), 
depthead int  references employee(empid));

INSERT INTO EMPDEPT
VALUES 
("E-101","HR","Monday",105),
("E-102","Development","Tuesday",101),
("E-103","HouseKeeping","Saturday",103),
("E-104","Sales","Sunday",104),
("E-105","Purchase","Tuesday",104);

create table empsalary(empid int  key references employee(empid), salary bigint, ispermanent varchar(3));

INSERT INTO EMPSALARY
VALUES
(101,2000,"Yes"),
(102,10000,"Yes"),
(103,5000,"No"),
(104,1900,"Yes"),
(105,2300,"Yes");

create table project(projectid varchar(50) primary key, duration int);

INSERT INTO PROJECT
VALUES
("p-1",23),
("p-2",15),
("p-3",45),
("p-4",2),
("p-5",30);

create table country(cid varchar(50) primary key, cname varchar(100));

INSERT INTO COUNTRY 
VALUES
("c-1","India"),
("c-2","USA"),
("c-3",	"China"),
("c-4",	"Pakistan"),
("c-5",	"Russia");

create table clienttable(clientid varchar(50) primary key, clientname varchar(100), cid varchar(50) references country(cid));

INSERT INTO CLIENTTABLE 
VALUES
("cl-1","ABC Group","c-1"),
("cl-2"	,"PQR","c-1"),
("cl-3","XYZ","c-2"),
("cl-4","tech altum","c-3"),
("cl-5"	,"mnp",	"c-5");

create table empproject(empid int  references employee(empid), projectid varchar(50)  references project(projectid),
 clientid varchar(50) references clienttable(clientid),startyear int, endyear int);
 
 INSERT INTO EMPPROJECT
 VALUES
 (101,"	p-1","	Cl-1",	2010,	2010),
(102,"p-2",	"Cl-2",	2010,2012),
(103,"p-1",	"Cl-3",2013	,NULL),
(104,"p-4","Cl-1",2014,2015),
(105,"p-4","Cl-5",2015,NULL);	

USE SQLPRACTICE;
-- 1.   Select the detail of the employee whose name start with P.

select *
FROM employee
where empname like "P%";

-- 2.   How many permanent candidate take salary more than 5000.

SELECT *
FROM employee;
SELECT *
from empdept;

-- 2.   How many permanent candidate take salary more than 5000.
select count(empid) as count
from empsalary
where ispermanent="Yes" and salary>5000;

-- 3.   Select the detail of employee whose emailId is in gmail.

select *
from employee
where emaildid like "%gmail%" ;

-- 4.   Select the details of the employee who work either for department E-104 or E-102.

select *
from employee
where department="E-104" or department= "E-102";

select *
from employee
where department in ("E-104","E-102");

-- 5.   What is the department name for DeptID E-102?

SELECT deptname
FROM empdept 
where DEPTID="E-102";

-- 6.  What is total salarythat is paid to permanent employees?

select *
from empsalary;

select sum(salary) as total
from empsalary
where ispermanent="Yes";

-- 7.  List name of all employees whose name ends with a.
select *
from employee
where empname like "%a";

-- 8.  List the number of department of employees in each project.
select *
from empproject;
select *
from employee;
select *
from empdept;
select count(department) no_dept
from employee;

select count(empid) as employee ,projectid
 from empproject 
 group by projectid; 

-- 9.  How many project started in year 2010.

select count(projectid) as total
from empproject
where startyear="2010";

-- 10.  How many project started and finished in the same year.

select count(projectid)
from empproject
where startyear=endyear;

-- 11.  select the name of the employee whose name's 3rd charactor is 'h'.

select empname
from employee
where empname like "__h%";

use sqlpractice;

-- Nested Queries
-- 1.   Select the department name of the company which is assigned to the employee whose employee id is grater 103.--
select * 
from employee;
select *
from empdept;

select deptname
from empdept
join employee
on empdept.deptid=employee.department
where empid>103 ;

select deptname from empdept
where deptid in 
(select department from employee where empid>103);

-- 2.   Select the name of the employee who is working under Abhishek.
select *
from employee;
select empname
from employee
where empheadid=105;

-- 3.   Select the name of the employee who is department head of HR.
select empname
from employee where empid=
(select depthead from empdept where deptname="HR");

















