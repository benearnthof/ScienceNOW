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


def chunk_list(a, n):
    """Wrapper to split list of docs a into n chunks of approximately equal length."""
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]
