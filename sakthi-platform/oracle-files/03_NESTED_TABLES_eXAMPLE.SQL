DECLARE
  -- Define a nested table type for employee IDs
  TYPE EmpIdTab IS TABLE OF NUMBER;
  -- Define a nested table type for salary increments
  TYPE SalIncTab IS TABLE OF NUMBER;

  -- Declare nested table variables
  emp_ids  EmpIdTab := EmpIdTab(101, 102, 103);
  sal_incs SalIncTab := SalIncTab(1000, 2000, 1500);

BEGIN
  -- Loop through the nested tables to update employee salaries
  FOR i IN 1 .. emp_ids.COUNT LOOP
    UPDATE employees
    SET salary = salary + sal_incs(i)
    WHERE employee_id = emp_ids(i);
  END LOOP;

  DBMS_OUTPUT.PUT_LINE('Salaries updated using Nested Table.');

END;
