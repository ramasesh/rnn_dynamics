import jax

print("JAX imported")

print("Devices")
print(jax.devices())

s = 1234
key = jax.random.PRNGKey(s)
print("successfully made key")

k0, k1, k2 = jax.random.split(key, 3)
print("successfully printed 3 keys")
