import unittest
from transformer import SQLTokenizer, SemanticTransformerParser

class TestTransformer(unittest.TestCase):
    def test_basic_query(self):
        schema = {
            "tables": [{"name": "customers", "columns": [{"name": "id"}, {"name": "name"}]}]
        }
        parser = SemanticTransformerParser(schema)
        res = parser.parse_to_sql("Show all customers", dialect="postgresql")
        self.assertIn("SELECT", res["sql"])

if __name__ == "__main__":
    unittest.main()
