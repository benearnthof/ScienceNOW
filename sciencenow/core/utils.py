"""
Utility functions.
"""

def get_plaintext_name(labs, taxonomy):
            """Will join multi class labels to one distinct label with `&`"""
            p_labels = []
            for item in labs:
                tmp = item.split(" ")
                plaintext_labels = [taxonomy[k] if k in taxonomy else "default" for k in tmp]
                plaintext_labels = " & ".join(plaintext_labels)
                p_labels.append(plaintext_labels)
            return p_labels