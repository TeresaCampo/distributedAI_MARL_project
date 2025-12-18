from pettingzoo.test import parallel_api_test
from myenv_independent_IQ import MyGridWorld

def test_my_env():
    env = MyGridWorld()

    try:
        parallel_api_test(env, num_cycles=1000)
        print("✅ Environment is compliant with PettingZoo API.")
    except Exception as e:
        print("❌ Test failed: environment is not compliant with PettingZoo API.")
        print(e)

if __name__ == "__main__":
    test_my_env()