/* Import data tables into the DB
    will have to develop script call from either bash or python */
    
.mode csv
.import /home/joseh/data/cadr_update/enrollments.csv enrollment
.schema enrollment

.mode csv
.import /home/joseh/data/cadr_update/Dim_Student.csv Dim_Student
.schema Dim_Student

.mode csv
.import /home/joseh/data/cadr_update/Dim_School.csv Dim_School
.schema Dim_School

.mode csv
.import /home/joseh/data/cadr_update/hsCourses.csv hsCourses
.schema hsCourses

.mode csv
.import /home/joseh/data/cadr_update/postSecDems.csv postSecDems
.schema postSecDems





