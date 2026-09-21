package com.avtracker.mobile.db

import java.sql.Connection
import java.sql.DriverManager
import java.sql.PreparedStatement

/** [SqlDatabase] over sqlite-jdbc, so the JVM tests run TrackerDb's SQL on a real SQLite engine. */
class JdbcSqlDatabase(private val conn: Connection) : SqlDatabase {

    init {
        conn.createStatement().use { it.execute("PRAGMA foreign_keys=ON") }
    }

    private fun bind(statement: PreparedStatement, args: List<Any?>) {
        args.forEachIndexed { i, v ->
            when (v) {
                null -> statement.setObject(i + 1, null)
                is Boolean -> statement.setLong(i + 1, if (v) 1 else 0)
                is Int -> statement.setLong(i + 1, v.toLong())
                is Float -> statement.setDouble(i + 1, v.toDouble())
                else -> statement.setObject(i + 1, v)
            }
        }
    }

    override fun execute(sql: String, args: List<Any?>) {
        conn.prepareStatement(sql).use { bind(it, args); it.executeUpdate() }
    }

    override fun insert(sql: String, args: List<Any?>): Long {
        execute(sql, args)
        conn.createStatement().use { st -> st.executeQuery("SELECT last_insert_rowid()").use { rs -> rs.next(); return rs.getLong(1) } }
    }

    override fun query(sql: String, args: List<Any?>): List<Map<String, Any?>> =
        conn.prepareStatement(sql).use { statement ->
            bind(statement, args)
            statement.executeQuery().use { rs ->
                val rows = ArrayList<Map<String, Any?>>()
                val meta = rs.metaData
                while (rs.next()) {
                    val row = LinkedHashMap<String, Any?>()
                    for (c in 1..meta.columnCount) {
                        row[meta.getColumnLabel(c)] = when (val v = rs.getObject(c)) {
                            is Int -> v.toLong()
                            is Float -> v.toDouble()
                            else -> v
                        }
                    }
                    rows += row
                }
                rows
            }
        }

    override fun executeScript(script: String) {
        for (statement in script.split(';')) if (statement.isNotBlank()) conn.createStatement().use { it.execute(statement) }
    }

    override fun <T> transaction(block: () -> T): T {
        val previous = conn.autoCommit
        conn.autoCommit = false
        try {
            val result = block()
            conn.commit()
            return result
        } catch (t: Throwable) {
            conn.rollback()
            throw t
        } finally {
            conn.autoCommit = previous
        }
    }

    override fun close() = conn.close()

    companion object {
        fun inMemory(): JdbcSqlDatabase = JdbcSqlDatabase(DriverManager.getConnection("jdbc:sqlite::memory:"))
    }
}
