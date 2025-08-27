find . -type f -name "*.cpp" -exec bash -c '
for file; do
    mv -- "$file" "${file%.cpp}.cu"
done
' _ {} +

find . -type f -name "*.hpp" -exec bash -c '
for file; do
    mv -- "$file" "${file%.hpp}.cuh"
done
' _ {} +
