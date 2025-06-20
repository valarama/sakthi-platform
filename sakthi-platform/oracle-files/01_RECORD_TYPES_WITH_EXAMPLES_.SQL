
--1. using rowtype attribute
DECLARE
  -- Declare a record that can hold an entire row from EMPLOYEES
  emp_rec employees%ROWTYPE;

BEGIN
  -- Select one employee's data into the record
  SELECT * INTO emp_rec
  FROM employees
  WHERE employee_id = 101;

  -- Display some fields using DBMS_OUTPUT
  DBMS_OUTPUT.PUT_LINE('Name: ' || emp_rec.first_name || ' ' || emp_rec.last_name);
  DBMS_OUTPUT.PUT_LINE('Salary: ' || emp_rec.salary);
END;


--2. Custom record type

DECLARE
  -- Create a custom record type
  TYPE EmpRecType IS RECORD (
    emp_id     employees.employee_id%TYPE,
    emp_name   employees.first_name%TYPE,
    emp_salary employees.salary%TYPE
  );

  -- Declare a variable of the custom record type
  emp_rec EmpRecType;

BEGIN
  -- Fetch specific fields into the custom record
  SELECT employee_id, first_name, salary INTO emp_rec
  FROM employees
  WHERE employee_id = 102;

  DBMS_OUTPUT.PUT_LINE('ID: ' || emp_rec.emp_id);
  DBMS_OUTPUT.PUT_LINE('Name: ' || emp_rec.emp_name);
  DBMS_OUTPUT.PUT_LINE('Salary: ' || emp_rec.emp_salary);
END;

--3. Using cursor for record types
DECLARE
  -- Cursor for selecting employees
  CURSOR emp_cur IS
    SELECT employee_id, salary FROM employees WHERE department_id = 60;

  -- Declare a record to hold each row from the cursor
  emp_rec emp_cur%ROWTYPE;

BEGIN
  OPEN emp_cur;
  LOOP
    FETCH emp_cur INTO emp_rec;
    EXIT WHEN emp_cur%NOTFOUND;

    -- Give a 10% raise
    UPDATE employees
    SET salary = emp_rec.salary * 1.10
    WHERE employee_id = emp_rec.employee_id;
  END LOOP;
  CLOSE emp_cur;

  DBMS_OUTPUT.PUT_LINE('Salaries updated for department 60.');
END;
