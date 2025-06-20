DECLARE
  -- Define an associative array (INDEX BY) for salary updates
  TYPE EmpSalaryTab IS TABLE OF NUMBER INDEX BY PLS_INTEGER;

  -- Declare the array variable
  emp_salary_increments EmpSalaryTab;

  -- Temporary variable
  v_emp_id NUMBER;

BEGIN
  -- Populate the associative array with employee IDs and their salary increments
  emp_salary_increments(101) := 1000;
  emp_salary_increments(102) := 1500;
  emp_salary_increments(103) := 2000;

  -- Loop through the array using FIRST..LAST
  v_emp_id := emp_salary_increments.FIRST;
  WHILE v_emp_id IS NOT NULL LOOP
    -- Update salary for each employee
    UPDATE employees
    SET salary = salary + emp_salary_increments(v_emp_id)
    WHERE employee_id = v_emp_id;

    v_emp_id := emp_salary_increments.NEXT(v_emp_id);
  END LOOP;

  DBMS_OUTPUT.PUT_LINE('Salaries updated using INDEX BY TABLE.');

END;
