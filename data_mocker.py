from langchain_core.messages import HumanMessage

class BillPayerMsgMocker():
    def __init__(self):
        self._message_with_context = {}

    def mock_input_message(self, msg):
        self._message_with_context.update({"input_message": msg})
        return self

    def mock_check_balance_response(self, query):
        self._message_with_context.update({"check_balance":query})
        return self

    def mock_electricity_bill_amount(self, response: str):
        self._message_with_context.update({"get_bill_amount": response})
        return self

    def mock_deduct_amount_response(self, response: str):
        self._message_with_context.update({"deduct_amount": response})
        return self

    def mock_electiricity_paying_response(self, response: str):
        self._message_with_context.update({"pay_electricity_bill": response})
        return self

    def mock_initial_balance_amount(self, amount):
        self._message_with_context.update({"initial_balance": amount})
        return self

    def construct_message(self):
        return self._message_with_context

def get_valid_messages():
    msg = BillPayerMsgMocker().mock_input_message(HumanMessage(content="Please pay my electricity bill."))\
                              .mock_check_balance_response({"msg":"Balance : 10000$", "balance":10000.0})\
                              .mock_electricity_bill_amount({"msg":"Electricity bill amount: 200$", "balance":10000.0})\
                              .mock_deduct_amount_response({"msg": "Deducted amount, new balance: 9900$", "balance":9900.0})\
                              .mock_electiricity_paying_response({"msg": "Connection failed. Maintainence. Try after 02/12/24", "balance": 10000.0})\
                              .mock_initial_balance_amount(10000.0)\
                              .construct_message()

    return [msg]


def get_node_vs_tools_mapping():
    return {"electricity_agent": ["check_balance", "get_bill_amount", "deduct_amount", "pay_electricity_bill"]}
