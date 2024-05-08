"""Main program to perform automate flux predictions flux with profet package."""

from profet.forecast.model01 import monthly_flux_model01


def main():
    """Execute flux predictions with PyTorch models."""

    monthly_flux_model01()

if __name__ == "__main__":

    main()
