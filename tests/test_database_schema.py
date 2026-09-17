import pytest

from stopsign.database import _VEHICLE_PASS_COLUMNS
from stopsign.database import Database
from stopsign.database import VehiclePassSchemaError


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _SchemaSession:
    def __init__(self, columns, *, fail_migration=False):
        self.columns = set(columns)
        self.fail_migration = fail_migration
        self.statements = []
        self.committed = False
        self.rolled_back = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, statement):
        sql = str(statement)
        self.statements.append(sql)
        if "information_schema.columns" in sql:
            return _Result([(column,) for column in sorted(self.columns)])
        if sql.startswith("ALTER TABLE"):
            if self.fail_migration:
                raise RuntimeError("permission denied")
            column = sql.split()[5]
            self.columns.add(column)
        return _Result([])

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def _database(session, *, read_only=False):
    database = Database.__new__(Database)
    database.Session = lambda: session
    database.read_only_mode = read_only
    return database


def test_missing_vehicle_pass_column_migration_failure_is_startup_error():
    session = _SchemaSession(_VEHICLE_PASS_COLUMNS.keys(), fail_migration=True)
    session.columns.remove("stream_lag_est_sec")

    with pytest.raises(VehiclePassSchemaError, match="stream_lag_est_sec"):
        _database(session)._ensure_vehicle_pass_columns()

    assert session.rolled_back is True


def test_complete_vehicle_pass_schema_passes_verification():
    session = _SchemaSession(_VEHICLE_PASS_COLUMNS.keys())

    _database(session)._ensure_vehicle_pass_columns()

    assert session.committed is False
    assert len([sql for sql in session.statements if sql.startswith("ALTER TABLE")]) == 0


def test_read_only_mode_rejects_missing_vehicle_pass_column():
    session = _SchemaSession(_VEHICLE_PASS_COLUMNS.keys())
    session.columns.remove("clip_status")

    with pytest.raises(VehiclePassSchemaError, match="clip_status"):
        _database(session, read_only=True)._ensure_vehicle_pass_columns()

    assert not any(sql.startswith("ALTER TABLE") for sql in session.statements)
