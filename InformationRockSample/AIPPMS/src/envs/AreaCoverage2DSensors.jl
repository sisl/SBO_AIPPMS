## SENSOR BLOCK

struct FarLidarSensor <: Sensor
    energy_cost::Float64
    max_fidelity::Float64
    fidelity_decay_rate::Float64
    effective_range::Float64
end

function FarLidarSensor()
    energy_cost = 1.0
    max_fidelity = 0.8
    fidelity_decay_rate = 0.96
    effective_range = -log(2*max_fidelity)/log(fidelity_decay_rate)
    return FarLidarSensor(energy_cost, max_fidelity, fidelity_decay_rate, effective_range)
end

struct NearLidarSensor <: Sensor
    energy_cost::Float64
    max_fidelity::Float64
    fidelity_decay_rate::Float64
    effective_range::Float64
end

function NearLidarSensor()
    energy_cost = 0.5
    max_fidelity = 0.8
    fidelity_decay_rate = 0.9
    effective_range = -log(2*max_fidelity)/log(fidelity_decay_rate)
    return NearLidarSensor(energy_cost, max_fidelity, fidelity_decay_rate, effective_range)
end

struct HDCamSensor <: Sensor
    energy_cost::Float64
    max_fidelity::Float64
    fidelity_decay_rate::Float64
    effective_range::Float64
end

function HDCamSensor()
    energy_cost = 1.5
    max_fidelity = 0.95
    fidelity_decay_rate = 0.9
    effective_range = -log(2*max_fidelity)/log(fidelity_decay_rate)
    return HDCamSensor(energy_cost, max_fidelity, fidelity_decay_rate, effective_range)
end

get_energy_cost(sensor::S) where {S <: Sensor} = sensor.energy_cost
