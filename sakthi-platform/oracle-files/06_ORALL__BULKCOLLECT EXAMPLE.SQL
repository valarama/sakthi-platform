DECLARE
  -- Constants
  c_limit CONSTANT PLS_INTEGER := 50;

  -- Collections
  TYPE EmpIdTab IS TABLE OF employees.employee_id%TYPE;
  TYPE SalaryTab IS TABLE OF employees.salary%TYPE;

  v_emp_ids  EmpIdTab;
  v_salaries SalaryTab;

  -- Cursor to select target employees
  CURSOR emp_cur IS
    SELECT employee_id, salary FROM employees WHERE department_id = 60;

  -- Loop control
  CURSOR_DONE BOOLEAN := FALSE;

BEGIN
  OPEN emp_cur;

  LOOP
    -- Fetch in batches using LIMIT
    FETCH emp_cur BULK COLLECT INTO v_emp_ids, v_salaries LIMIT c_limit;

    -- Exit if no more rows
    EXIT WHEN v_emp_ids.COUNT = 0;

    BEGIN
      -- Perform bulk update using FORALL with SAVE EXCEPTIONS
      FORALL i IN 1 .. v_emp_ids.COUNT SAVE EXCEPTIONS
        UPDATE employees
        SET salary = v_salaries(i) * 1.10
        WHERE employee_id = v_emp_ids(i);

    EXCEPTION
      WHEN OTHERS THEN
        -- Handle and display SQL%BULK_EXCEPTIONS
        FOR j IN 1 .. SQL%BULK_EXCEPTIONS.COUNT LOOP
          DBMS_OUTPUT.PUT_LINE('Error at index ' || SQL%BULK_EXCEPTIONS(j).ERROR_INDEX ||
                               ': Code=' || SQL%BULK_EXCEPTIONS(j).ERROR_CODE);
        END LOOP;
    END;
  END LOOP;

  CLOSE emp_cur;

  DBMS_OUTPUT.PUT_LINE('Bulk salary update completed with exception handling.');

END;
