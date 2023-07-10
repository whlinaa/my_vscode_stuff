use college;
-- find those clients who have made a purchase
select name, client_id
from clients
where client_id in 
(
select client_id
from invoices
)
order by client_id;

-- find those invoices with invoice total > 120
select invoice_total, client_id
from invoices
where invoice_total>120;

-- count how many customers place an order
select count(distinct client_id) 
from invoices;

-- count total records in invoices
select count(*) 
from invoices;

-- find all client names who make > 1 purchases
select c.name, count(*) as n_purchases
from clients c, invoices i
where c.client_id = i.client_id
group by c.client_id
having n_purchases > 1;

use sql_invoicing;
-- show those who make >=2 purchases. Also show their total spending
select client_id, count(client_id) as n_purchases, sum(invoice_total) as total_spending
from invoices
group by client_id
having n_purchases >= 2
order by total_spending desc;


select name
from clients
where name like "%t%"; -- contain t word


select count(*)
from invoices
where payment_date is null;



-- sql_hr table
-- show the count for each job title
select job_title, count(*)
from employees
group by job_title
order by job_title;

-- find those who doesn't have a supervisor
select first_name, last_name
from employees
where reports_to is NULL;

-- find the number of people each employee supervisess
-- TODO
select last_name, count(*)
from employees
group by reports_to;


-- find the manager of each employee
select e1.first_name, e2.first_name as manager
from employees e1, employees e2
where e1.reports_to = e2.employee_id;

-- find the employee whose salary > their manager
select e1.first_name, e2.first_name as manager, e1.salary as testing, e2.salary
from employees e1, employees e2
where e1.reports_to = e2.employee_id and e1.salary > e2.salary;


-------------------------
-- customers table in sql_store database
use sql_store;
-- classifier customer by points
-- <2000 => bronze, [2000,3000) silver, >3000 => gold, 
SELECT customer_id, first_name, points, 'Bronze' as 'type'
FROM customers
where points<2000
Union
SELECT customer_id, first_name, points, 'Silver' as 'type'
FROM customers
where points>=2000 and points <3000 -- or where points between 2000 and 2999
Union
SELECT customer_id, first_name, points, 'Gold' as 'type'
FROM customers
where points>=3000
order by first_name -- just need this once
;

use college;
---- create a db about students
drop database if exists college;
drop table if exists students;
drop table if exists enrolments;
drop table if exists courses;

create database college;
use college;

CREATE TABLE student
(
sid varCHAR(20) NOT NULL,
name varCHAR(20) NOT NULL,
age int(10) NOT NULL check(age between 0 and 150),
gpa decimal(9,2) NOT NULL check(gpa > 0),
primary key (sid)
);

drop table students;

use college;

rename table student to students; -- rename

alter table students add (gender varchar(10)); -- add
select * from students;
-- alter table students modify (gender varchar(20)); -- modify an existing column
alter table students drop (gender); -- delete a column 
---- DML

select * from students;

--- insert
insert into students(sid, name, age, gpa) values 
(1, 'Joe', 18, 3.14), 
(2, 'Mary', 18, 2.14), 
(3, 'Sally', 19, 3.24), 
(4, 'Tom', 19, 3.11), 
(5, 'John', 17, 3.34),
(6, 'Wong', 17, 3.14),
(7, 'Wang', 17, 1.2);
---- delete
delete from students where sid =3;
select * from students;
delete  from students;  --- delete all 
select * from students;
update students set gpa = gpa + 0.1 where gpa <2.5;
CREATE TABLE courses
(
cid varCHAR(20) NOT NULL,
name varCHAR(20) NOT NULL,
dept varchar(20) NOT NULL,
primary key (cid)
);

insert into students value(8,'Alisa',17,1.9);

use college;

insert into courses(cid, name, dept) values 
(1, 'calculus', 'math'), 
(2, 'OS', 'comp'), 
(3, 'discrete math', 'comp'), 
(4, 'data science', 'comp'), 
(5, 'english', 'LANG');
select * from courses;

drop table enrolments;
CREATE TABLE enrolments
(
sid varCHAR(20) NOT NULL, -- references students(sid)
cid varchar(20) not null, -- references courses(cid)
score decimal(9,2) not null,
primary key (sid, cid),
foreign key (sid) references students(sid),
foreign key (cid) references courses(cid)
);

insert into enrolments(sid, cid, score) values
(1, 1, 50),
(1, 2, 40), 
(1, 3, 30), 
(1, 4, 20), 
(1, 5, 100), 

(2, 1, 90), 
(2, 2, 10), 
(2, 4, 20), 
(2, 5, 95),

(3, 1, 70), 
(3, 3, 60), 
(3, 4, 20), 
(3, 5, 95),

(4, 1, 50), 
(4, 2, 80),

(5, 1, 90), 
(5, 2, 90), 
(5, 4, 95), 
(5, 5, 77),

(6, 1, 90), 
(6, 2, 90), 
(6, 3, 10), 
(6, 5, 77),

(7, 1, 90), 
(7, 2, 0), 
(7, 4, 5), 
(7, 5, 77),

(8, 1, 9), 
(8, 2, 90), 
(8, 3, 95), 
(8, 5, 7);

delete from enrolments;

select * from enrolments;
alter table enrolments add constraint check (score between 0 and 100);

select * from students;
select * from courses;
select * from enrolments;

-- COMP exam questions
-- get those COMP courses with enrolment > 1
select  c.name, count(*) as total
from courses c, enrolments e
where c.cid = e.cid and c.dept='COMP'
group by c.name
having total>1
order by total DESC;

-- for each dept course, find the course with the most enrollment
-- TO DO
select c.name, c.dept, count(*) as total
from courses c, enrolments e
where c.cid = e.cid 
group by c.name, c.dept
having total>1
order by total DESC;

-- show the scores of the student with the highest gpa 
select c.name, e.score
from students s, courses c, enrolments e
where s.sid = e.sid and c.cid = e.cid 
and s.gpa = (
	select max(gpa)
	from students
)
order by e.score desc;

-- deduct the students in course 1 by the sd od that course
select * from enrolments
order by cid;
update enrolments set score = score - 
(
	select * from (
select stddev(score)
from enrolments
where cid = 1
	) as temp
)
where cid = 1;
-------------


-- gpa group by age
select age, avg(gpa)
from students
group by age;
-- 

-- show the courses taken by the students 
select s.name as `student name`, c.name as 'course name', e.score as 'score'
from courses c, enrolments e, students s
where c.cid = e.cid and s.sid = e.sid and s.gpa>3.0
order by `student name` desc;


-- show the # of courses taken by each student
select s.sid, s.name, s.gpa, count(*) as '# courses taken'
from courses c, enrolments e, students s
where c.cid = e.cid and s.sid = e.sid 
group by s.sid
order by s.gpa desc ;





-- find the students who took the most courses. 
-- M1: This is no good, since it returns only one entry. If there are duplicates (i.e., two students who took equal number of courses, we'll miss it.)s
select s.sid, s.name, s.gpa, count(*) as '# courses taken'
from courses c, enrolments e, students s
where c.cid = e.cid and s.sid = e.sid 
group by s.sid
order by count(*) desc 
limit 1;
-- show only those students who took the most courses
-- M2: show all 
select s.sid, s.name, s.gpa, count(*) as '# courses taken'
from courses c, enrolments e, students s
where c.cid = e.cid and s.sid = e.sid
group by s.sid
having count(*) =
( 	select count(*)
	from enrolments
	group by sid
	order by count(*) desc
	limit 1 
) 
order by s.gpa desc;

-- show each course's details. Show its mean score and standard deviation
select c.name, count(*) as `students enrolment`, avg(e.score) as `mean score`, stddev(e.score) as `standard deviation`
from enrolments e, courses c
where e.cid = c.cid 
group by c.cid 
order by `mean score` desc; -- must use backtick...

-- show the avg score of students with score >30 for each course, and then show only course with > 3 enrolments
select cid, avg(score)
from enrolments
where score>30
group by cid
having count(*)>3;


--- if there are NULL, count() in that column only counts non-NULL values!
select count(*), count(gender)
from students;

create view students_details as
(
	select sid, name
	from students
);

-- from those students whose gpa is great than at least one of the students of age 18. That is, larger than the lowest gpa studnet of age 18
-- M1
select distinct s1.name
from students s1, students s2
where s1.gpa > s2.gpa and s2.age = 18;

-- M2. Find the minimum first
select s.name
from students s
where s.gpa > (
select  gpa
from students 
where age = 18
order by gpa
limit 1
);

-- M3. Use `some`
select s.name
from students s
where s.gpa > some(
select  gpa
from students 
where age = 18
);


-- get all attributes of s
select s.*, e.score
from students s, enrolments e
where s.sid = e.sid;

-- tuple comparsion
select *
from students 
where (age, gpa)>(17,3.1); -- same as age >17 and gpa >3.1

-- set operations. duplicates will be automatically removed!!!
-- if we want to retain duplicates, use `union all` instead of `union`
(select  cid
from enrolments
where score > 90)
union -- use `union all` in want to 
(select  cid
from courses
where dept = 'comp')
order by cid;

-- show average score of each student in his/her courses 
select s.name, avg(score) as `average student score`
from enrolments e, students s 
where e.sid = s.sid
group by s.sid
order by `average student score` desc;

-- find the total # of students using the enrolment table only
-- trick: use `distinct` inside count() function!!
select count(distinct sid)
from enrolments;
select avg(distinct score)
from enrolments;
-- find students who took course 'data science'
-- M1
select name 
from students s
where sid in 
(
	select sid
	from enrolments e, courses c
	where e.cid = c.cid and c.name ='data science'
);

-- M2
select s.name
from courses c, enrolments e, students s
where c.cid = e.cid and s.sid = e.sid and c.name = 'data science';

-- find highest average score of course
select cid, avg(score)
from enrolments
group by cid
having avg(score) >= all
(
	select avg(score)
	from enrolments
	group by cid
);
-- M2
select cid, avg(score)
from enrolments e
group by cid
order by avg(score) desc
limit 1;

-- find the students who took both course 3 and course 4 
select sid
from enrolments e1
where cid = 3 and exists
(
	select *
	from enrolments e2
	where cid = 4 and e1.sid = e2.sid
);

-- find students' average score of those with average > 50
select sid, avg(score)
from enrolments
group by sid
having avg(score)>50;

-- M2. do away with `having` clause
select sid, avg_score
from (
	select sid, avg(score) as avg_score
	from enrolments
	group by sid
	) as temp
where avg_score>50;

-- find maximum total scores
select cid, max(total_score)
from
(
select cid, sum(score) as total_score
from enrolments
group by cid
) as temp;
-- 

-- In student table, for each age group, find the maximum gpa/sid ratio. Return name
select name, age, max(gpa/sid)
from students
group by age
having age>17
order by age desc;

--
insert into students
values('9', 'Henry', 30, 4.1), (10, 'Sam', 25, 4.2);

update students
set gpa = 4.3, age = 20
where name = 'Sam';


alter table students modify name varchar(30);

delete from students
where name = 'Sam';


drop table teachers;

create table teachers
(
	tid int NOT NULL,
	name varchar(20),
	age int(10),
	salary int(10),
	supervisor int(10),
	primary key(tid)
);

delete from teachers;

insert into teachers
values (1, 'Raymond Wong', 40, 60, 2), (2, 'Nevin', 50, 70, 3);


-- find all duplicates
select name, count(*)
from students
group by name
having count(*)>1;

-- select students whose age in his age group with minimum sid. That is, remove duplicates
select name, age, sid
from students 
where sid not in 
(
select s1.sid
from students s1, students s2
where s1.age = s2.age and s1.sid > s2.sid
);




select name
from students
where age>18 and age>3.0
limit 1 offset 2;


select * from people;

insert into people VALUES(2,'Peter');
insert into people VALUES(3,'P1');
insert into people VALUES(4,'P2');
insert into people VALUES(5,'P3');
insert into people VALUES(6,'P4');
insert into people VALUES(7,'P5');


-- create Table customer 
CREATE TABLE student9
(
sid varCHAR(20) NOT NULL,
name varCHAR(20) NOT NULL,
-- age int(10) NOT NULL check(age between 0 and 150),
age int(10),
gpa decimal(9,2) NOT NULL check(gpa > 0),
primary key (sid)
);


CREATE TABLE courses
(
cid varCHAR(20) NOT NULL,
name varCHAR(20) NOT NULL,
dept varchar(20) NOT NULL,
primary key (cid)
);