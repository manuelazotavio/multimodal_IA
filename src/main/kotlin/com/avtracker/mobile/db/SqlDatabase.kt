package com.avtracker.mobile.db

import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import java.io.File

/**
 * The little of SQL access [TrackerDb] needs. Values are `Long`, `Double`, `String`, `ByteArray` or null, like Python's
 * sqlite3. The app uses Android's SQLite ([AndroidSqlDatabase]); the JVM tests run the same SQL on sqlite-jdbc.
 */
interface SqlDatabase : AutoCloseable {
    fun execute(sql: String, args: List<Any?> = emptyList())

    /** Runs an INSERT and returns the new row id (Python `cursor.lastrowid`). */
    fun insert(sql: String, args: List<Any?> = emptyList()): Long

    fun query(sql: String, args: List<Any?> = emptyList()): List<Map<String, Any?>>

    /** Several statements separated by ';' (Python `executescript`). */
    fun executeScript(script: String)

    /** One transaction: committed if [block] returns, rolled back if it throws (Python's `_cursor` context manager). */
    fun <T> transaction(block: () -> T): T
}

class AndroidSqlDatabase(private val db: SQLiteDatabase) : SqlDatabase {

    override fun execute(sql: String, args: List<Any?>) {
        if (args.isEmpty()) db.execSQL(sql) else db.execSQL(sql, args.toTypedArray())
    }

    override fun insert(sql: String, args: List<Any?>): Long {
        val statement = db.compileStatement(sql)
        try {
            args.forEachIndexed { i, v ->
                val index = i + 1
                when (v) {
                    null -> statement.bindNull(index)
                    is Long -> statement.bindLong(index, v)
                    is Int -> statement.bindLong(index, v.toLong())
                    is Boolean -> statement.bindLong(index, if (v) 1 else 0)
                    is Double -> statement.bindDouble(index, v)
                    is Float -> statement.bindDouble(index, v.toDouble())
                    is ByteArray -> statement.bindBlob(index, v)
                    else -> statement.bindString(index, v.toString())
                }
            }
            return statement.executeInsert()
        } finally {
            statement.close()
        }
    }

    override fun query(sql: String, args: List<Any?>): List<Map<String, Any?>> {
        // rawQuery binds strings only; SQLite's column affinity converts them for the numeric comparisons used here.
        db.rawQuery(sql, args.map { it?.toString() }.toTypedArray()).use { cursor ->
            val rows = ArrayList<Map<String, Any?>>()
            while (cursor.moveToNext()) {
                val row = LinkedHashMap<String, Any?>()
                for (c in 0 until cursor.columnCount) {
                    row[cursor.getColumnName(c)] = when (cursor.getType(c)) {
                        Cursor.FIELD_TYPE_INTEGER -> cursor.getLong(c)
                        Cursor.FIELD_TYPE_FLOAT -> cursor.getDouble(c)
                        Cursor.FIELD_TYPE_STRING -> cursor.getString(c)
                        Cursor.FIELD_TYPE_BLOB -> cursor.getBlob(c)
                        else -> null
                    }
                }
                rows += row
            }
            return rows
        }
    }

    override fun executeScript(script: String) {
        for (statement in script.split(';')) if (statement.isNotBlank()) db.execSQL(statement)
    }

    override fun <T> transaction(block: () -> T): T {
        db.beginTransaction()
        try {
            val result = block()
            db.setTransactionSuccessful()
            return result
        } finally {
            db.endTransaction()
        }
    }

    override fun close() = db.close()

    companion object {
        /** `av_tracker.db` in the app's private storage: write-ahead log and foreign keys on, as `TrackerDB._get_conn` sets. */
        fun open(context: Context, name: String = "av_tracker.db"): AndroidSqlDatabase {
            val db = SQLiteDatabase.openOrCreateDatabase(File(context.filesDir, name), null)
            db.enableWriteAheadLogging()
            db.setForeignKeyConstraintsEnabled(true)
            return AndroidSqlDatabase(db)
        }
    }
}
