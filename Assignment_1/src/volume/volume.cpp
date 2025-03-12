#include "volume.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cctype> // isspace
#include <chrono>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <gsl/span>
#include <iostream>
#include <string>

struct Header {
    glm::ivec3 dim;
    size_t elementSize;
};
static Header readHeader(std::ifstream& ifs, const volume::FileExtension& fileExtension);
static Header readVolumeHeader_fld(std::ifstream& ifs);
static Header readVolumeHeader_dat(std::ifstream& ifs);

static float computeMinimum(gsl::span<const float> data);
static float computeMaximum(gsl::span<const float> data);
static std::vector<int> computeHistogram(gsl::span<const float> data);

namespace volume {

Volume::Volume(const std::filesystem::path& file)
    : m_fileName(file.string())
{
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    loadFile(file);
    auto end = clock::now();
    std::cout << "Time to load: " << std::chrono::duration<double, std::milli>(end - start).count() << "ms" << std::endl;

    if (m_data.size() > 0) {
        m_minimum = computeMinimum(m_data);
        m_maximum = computeMaximum(m_data);
        m_histogram = computeHistogram(m_data);
    }
}

Volume::Volume(std::vector<float> data, const glm::ivec3& dim)
    : m_fileName()
    , m_elementSize(2)
    , m_dim(dim)
    , m_data(std::move(data))
    , m_minimum(computeMinimum(m_data))
    , m_maximum(computeMaximum(m_data))
    , m_histogram(computeHistogram(m_data))
{
}

float Volume::minimum() const
{
    return m_minimum;
}

float Volume::maximum() const
{
    return m_maximum;
}

std::vector<int> Volume::histogram() const
{
    return m_histogram;
}

glm::ivec3 Volume::dims() const
{
    return m_dim;
}

std::string_view Volume::fileName() const
{
    return m_fileName;
}

int reflectIndex(int idx, int maxIdx)
{
    if (idx < 0)
        return -idx; // Reflect negative index.
    else if (idx > maxIdx)
        return 2 * maxIdx - idx; // Reflect indices above the maximum.
    return idx;
}


float Volume::getVoxel(int x, int y, int z) const
{
    x = reflectIndex(x, m_dim.x - 1);
    y = reflectIndex(y, m_dim.y - 1);
    z = reflectIndex(z, m_dim.z - 1);


    const size_t i = size_t(x + m_dim.x * (y + m_dim.y * z));
    return static_cast<float>(m_data[i]);
}

// This function returns a value based on the current interpolation mode
float Volume::getSampleInterpolate(const glm::vec3& coord) const
{
    switch (interpolationMode) {
    case InterpolationMode::NearestNeighbour: {
        return getSampleNearestNeighbourInterpolation(coord);
    }
    case InterpolationMode::Linear: {
        return getSampleTriLinearInterpolation(coord);
    }
    case InterpolationMode::Cubic: {
        return getSampleTriCubicInterpolation(coord);
    }
    default: {
        throw std::exception();
    }
    }
}

// This function returns the nearest neighbour value at the continuous 3D position given by coord.
// Notice that in this framework we assume that the distance between neighbouring voxels is 1 in all directions
float Volume::getSampleNearestNeighbourInterpolation(const glm::vec3& coord) const
{
    // check if the coordinate is within volume boundaries, since we only look at direct neighbours we only need to check within 0.5
    if (glm::any(glm::lessThan(coord + 0.5f, glm::vec3(0))) || glm::any(glm::greaterThanEqual(coord + 0.5f, glm::vec3(m_dim))))
        return 0.0f;

    // nearest neighbour simply rounds to the closest voxel positions
    auto roundToPositiveInt = [](float f) {
        // rounding is equal to adding 0.5 and cutting off the fractional part
        return static_cast<int>(f + 0.5f);
    };

    return getVoxel(roundToPositiveInt(coord.x), roundToPositiveInt(coord.y), roundToPositiveInt(coord.z));
}

// ======= TODO : IMPLEMENT the functions below for tri-linear interpolation ========
// ======= Consider using the linearInterpolate and biLinearInterpolate functions ===
// This function returns the trilinear interpolated value at the continuous 3D position given by coord.
float Volume::getSampleTriLinearInterpolation(const glm::vec3& coord) const
{
    int x0 = floor(coord.x);
    int y0 = floor(coord.y);
    int z0 = floor(coord.z);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    // check if the coordinate is within volume boundaries
    if (x0 < 0 || y0 < 0 || z0 < 0 || x1 >= m_dim.x || y1 >= m_dim.y || z1 >= m_dim.z)
        return 0.0f;

    // get the voxels
    float v000 = getVoxel(x0, y0, z0);
    float v001 = getVoxel(x0, y0, z1);
    float v010 = getVoxel(x0, y1, z0);
    float v011 = getVoxel(x0, y1, z1);
    float v100 = getVoxel(x1, y0, z0);
    float v101 = getVoxel(x1, y0, z1);
    float v110 = getVoxel(x1, y1, z0);
    float v111 = getVoxel(x1, y1, z1);

    // interpolate in the x direction, then y direction and z direction
    float i00 = linearInterpolate(v000, v100, coord.x - x0);
    float i01 = linearInterpolate(v001, v101, coord.x - x0);
    float i10 = linearInterpolate(v010, v110, coord.x - x0);
    float i11 = linearInterpolate(v011, v111, coord.x - x0);

    float i0 = linearInterpolate(i00, i10, coord.y - y0);
    float i1 = linearInterpolate(i01, i11, coord.y - y0);

    return linearInterpolate(i0, i1, coord.z - z0);
}

// This function linearly interpolates the value at X using incoming values g0 and g1 given a factor (equal to the positon of x in 1D)
//
// g0--X--------g1
//   factor
float Volume::linearInterpolate(float g0, float g1, float factor)
{   
    // linear interpolation between 2 values is the weighted sum between them
    // this is equivalent to the formula: f(g0) * (1 - |factor - 0|) + f(g1) * (1 - |factor - 1|)
    return (1 - factor) * g0 + factor * g1;
}

// This function bi-linearly interpolates the value at the given continuous 2D XY coordinate for a fixed integer z coordinate.
float Volume::biLinearInterpolate(const glm::vec2& xyCoord, int z) const
{
    int x0 = floor(xyCoord.x);
    int y0 = floor(xyCoord.y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // check if the coordinate is within volume boundaries
    if (x0 < 0 || y0 < 0 || x1 >= m_dim.x || y1 >= m_dim.y)
        return 0.0f;

    // get the voxels
    float v00 = getVoxel(x0, y0, z);
    float v01 = getVoxel(x0, y1, z);
    float v10 = getVoxel(x1, y0, z);
    float v11 = getVoxel(x1, y1, z);

    // interpolate in the x direction and then in the y direction
    float i0 = linearInterpolate(v00, v10, xyCoord.x - x0);
    float i1 = linearInterpolate(v01, v11, xyCoord.x - x0);

    return linearInterpolate(i0, i1, xyCoord.y - y0);
}


// ======= OPTIONAL : This functions can be used to implement cubic interpolation ========
// This function represents the h(x) function, which returns the weight of the cubic interpolation kernel for a given position x

// As per https://en.wikipedia.org/wiki/Bicubic_interpolation 
float Volume::weight(float x)
{
    float alpha = -0.5;

    float abs_x = abs(x);

    if (abs_x <= 1)
    {
        return (alpha + 2) * (abs_x * abs_x * abs_x) - (alpha + 3) * (abs_x * abs_x) + 1;
    }
    else if (abs_x < 2)
    {
        return (alpha) * (abs_x * abs_x * abs_x) - 5 * (alpha) * (abs_x * abs_x) + 8 * (alpha) * (abs_x) - 4 * alpha;
    } else {
        return 0;
    }

    return 0;
}

// ======= OPTIONAL : This functions can be used to implement cubic interpolation ========
// This functions returns the results of a cubic interpolation using 4 values and a factor
float Volume::cubicInterpolate(float g0, float g1, float g2, float g3, float factor)
{
    // C0 - C1 - SamplePos - C2 - C3
    return g0 * weight(factor + 1.0f) + g1 * weight(factor) + g2 * weight(factor - 1.0f) + g3 * weight(factor - 2.0f);
}

// ======= OPTIONAL : This functions can be used to implement cubic interpolation ========
// This function returns the value of a bicubic interpolation
float Volume::biCubicInterpolate(const glm::vec2& xyCoord, int z) const
{
    // Determine the base coordinates and fractional offsets:
    int x = static_cast<int>(std::floor(xyCoord.x));
    int y = static_cast<int>(std::floor(xyCoord.y));
    float dx = xyCoord.x - x;
    float dy = xyCoord.y - y;
    // Interpolate along the x-axis for 4 rows in the neighborhood:
    float col[4];
    for (int j = -1; j <= 2; j++) {
        float row[4];
        for (int i = -1; i <= 2; i++) {
            // Retrieve voxel values from the volume.
            row[i + 1] = getVoxel(x + i, y + j, z);
        }
        col[j + 1] = cubicInterpolate(row[0], row[1], row[2], row[3], dx);
    }

    // Interpolate along the y-axis using the weighted contributions:
    return cubicInterpolate(col[0], col[1], col[2], col[3], dy);
}

// ======= OPTIONAL : This functions can be used to implement cubic interpolation ========
// This function computes the tricubic interpolation at coord
float Volume::getSampleTriCubicInterpolation(const glm::vec3& coord) const
{
    // Determine the base coordinates and fractional offsets:
    int x = static_cast<int>(std::floor(coord.x));
    int y = static_cast<int>(std::floor(coord.y));
    int z = static_cast<int>(std::floor(coord.z));
    float dx = coord.x - x;
    float dy = coord.y - y;


    float dz = coord.z - z;

    // Perform bicubic interpolation on 4 slices (z direction):
    float slab[4];
    for (int k = -1; k <= 2; k++) {
        // Bug when z == 0 
        slab[k + 1] = biCubicInterpolate(glm::vec2(coord.x, coord.y), z + k);
    }

    // Interpolate along the z-axis using the weight-based cubic interpolation:
    return cubicInterpolate(slab[0], slab[1], slab[2], slab[3], dz);
}

// Load an fld volume data file
// First read and parse the header, then the volume data can be directly converted from bytes to uint16_ts
void Volume::loadFile(const std::filesystem::path& file)
{
    assert(std::filesystem::exists(file));
    std::ifstream ifs(file, std::ios::binary);
    assert(ifs.is_open());

    // Normalize file extension to lowercase
    std::string extension = file.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    // Check file type
    if (extension == ".fld") {
        m_fileExtension = FileExtension::FLD;
    }
    else if (extension == ".dat") {
        m_fileExtension = FileExtension::DAT;
    }
    else {
        std::cerr << "Unsupported file extension: " << extension << "\n";
        return;
    }

    const auto header = readHeader(ifs, m_fileExtension);
    m_dim = header.dim;
    m_elementSize = header.elementSize;

    loadVolumeData(ifs);
}

void Volume::loadVolumeData(std::ifstream& ifs)
{
    const size_t voxelCount = static_cast<size_t>(m_dim.x * m_dim.y * m_dim.z);
    const size_t byteCount = voxelCount * m_elementSize;
    std::vector<char> buffer(byteCount);
    // Data section is separated from header by two /f characters.
    if (m_fileExtension == FileExtension::FLD) ifs.seekg(2, std::ios::cur);
    ifs.read(buffer.data(), std::streamsize(byteCount));

    m_data.resize(voxelCount);
    if (m_elementSize == 1) { // Bytes.
        for (size_t i = 0; i < byteCount; i++) {
            m_data[i] = static_cast<float>(buffer[i] & 0xFF);
        }
    } else if (m_elementSize == 2) { // uint16_ts.
        for (size_t i = 0; i < byteCount; i += 2) {
            m_data[i / 2] = static_cast<float>((buffer[i] & 0xFF) + (buffer[i + 1] & 0xFF) * 256);
        }
    }
}
}

static Header readHeader(std::ifstream& ifs, const volume::FileExtension& fileExtension)
{
    if (fileExtension == volume::FileExtension::FLD) {
        return readVolumeHeader_fld(ifs);
    } else if (fileExtension == volume::FileExtension::DAT) {
        return readVolumeHeader_dat(ifs);
    } else {
        return {};
    }
}

Header readVolumeHeader_dat(std::ifstream& ifs)
{
    Header out {};

    unsigned short sizeX, sizeY, sizeZ;
    const std::size_t nbytes = 2;
    char buff[nbytes];
    ifs.read(buff, nbytes);
    std::memcpy(&sizeX, buff, sizeof(int));
    ifs.read(buff, nbytes);
    std::memcpy(&sizeY, buff, sizeof(int));
    ifs.read(buff, nbytes);
    std::memcpy(&sizeZ, buff, sizeof(int));
    out.dim.x = sizeX;
    out.dim.y = sizeY;
    out.dim.z = sizeZ;
    out.elementSize = 2;
    return out;
}

Header readVolumeHeader_fld(std::ifstream& ifs)
{
    Header out {};

    // Read input until the data section starts.
    std::string line;
    while (ifs.peek() != '\f' && !ifs.eof()) {
        std::getline(ifs, line);
        // Remove comments.
        line = line.substr(0, line.find('#'));
        // Remove any spaces from the string.
        // https://stackoverflow.com/questions/83439/remove-spaces-from-stdstring-in-c
        line.erase(std::remove_if(std::begin(line), std::end(line), ::isspace), std::end(line));
        if (line.empty())
            continue;

        const auto separator = line.find('=');
        const auto key = line.substr(0, separator);
        const auto value = line.substr(separator + 1);

        if (key == "ndim") {
            if (std::stoi(value) != 3) {
                std::cout << "Only 3D files supported\n";
            }
        } else if (key == "dim1") {
            out.dim.x = std::stoi(value);
        } else if (key == "dim2") {
            out.dim.y = std::stoi(value);
        } else if (key == "dim3") {
            out.dim.z = std::stoi(value);
        } else if (key == "nspace") {
        } else if (key == "veclen") {
            if (std::stoi(value) != 1)
                std::cerr << "Only scalar m_data are supported" << std::endl;
        } else if (key == "data") {
            if (value == "byte") {
                out.elementSize = 1;
            } else if (value == "short") {
                out.elementSize = 2;
            } else {
                std::cerr << "Data type " << value << " not recognized" << std::endl;
            }
        } else if (key == "field") {
            if (value != "uniform")
                std::cerr << "Only uniform m_data are supported" << std::endl;
        } else if (key == "#") {
            // Comment.
        } else {
            std::cerr << "Invalid AVS keyword " << key << " in file" << std::endl;
        }
    }
    return out;
}

static float computeMinimum(gsl::span<const float> data)
{
    return float(*std::min_element(std::begin(data), std::end(data)));
}

static float computeMaximum(gsl::span<const float> data)
{
    return float(*std::max_element(std::begin(data), std::end(data)));
}

static std::vector<int> computeHistogram(gsl::span<const float> data)
{
    std::vector<int> histogram(size_t(*std::max_element(std::begin(data), std::end(data)) + 1), 0);
    for (const auto v : data)
        histogram[v]++;
    return histogram;
}
